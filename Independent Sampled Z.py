
# INDEPENDENT STRUCTURE SAMPLED-Z SENSITIVITY ANALYSIS

import warnings

warnings.filterwarnings(
    "ignore",
    message="Loop fusion failed because the resulting node would exceed the kernel argument limit.*",
    category=UserWarning
)

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

try:
    import pytensor.tensor as pt
except ImportError:
    import aesara.tensor as pt


DIRECT_DATA = r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\direct_dataset_simulated.csv"

COLLATERAL_DATA = r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\collateral_dataset_simulated.csv"

OUTDIR = Path(
    r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\Main Sensitivity\Independent_sampledZ_results"
)

TRACE_DIR = OUTDIR / "posterior_traces"
TRACEPLOT_DIR = OUTDIR / "traceplots"
SENSITIVITY_PLOT_DIR = OUTDIR / "sensitivity_plots"
TABLE_DIR = OUTDIR / "comparison_tables"

DIRECT_COLS = ["Y_t1", "Y_t2", "Y_t3"]
COLLATERAL_COLS = ["C_t1", "C_t2", "C_t3"]

N_DIRECT = 150
N_COLLATERAL = 200

MU_PRIOR_SDS = [2, 5, 10]

SIGMA_PRIOR_SPECS = [
    {
        "name": "HalfCauchy_beta1",
        "dist": "HalfCauchy",
        "kwargs": {"beta": 1.0}
    },
    {
        "name": "HalfCauchy_beta2p5",
        "dist": "HalfCauchy",
        "kwargs": {"beta": 2.5}
    },
    {
        "name": "HalfCauchy_beta5",
        "dist": "HalfCauchy",
        "kwargs": {"beta": 5.0}
    },
    {
        "name": "HalfNormal_sigma2",
        "dist": "HalfNormal",
        "kwargs": {"sigma": 2.0}
    }
]

Z_PRIOR_SPECS = [
    {
        "name": "Beta_1_1_constrained",
        "alpha": 1.0,
        "beta": 1.0
    },
    {
        "name": "Beta_2_2_constrained",
        "alpha": 2.0,
        "beta": 2.0
    }
]

Z_LOWER = 0.05
Z_UPPER = 0.95


DRAWS = 1000
TUNE = 1200
CHAINS = 4
CORES = 1

TARGET_ACCEPT = 0.97
RANDOM_SEED = 123


TRUE_VALUES = {
    "mu_omega": 0.50,
    "sigma_omega": 1.00,
    "sigma_direct": 1.20,
    "sigma_collateral": 0.80
}

PARAMETER_VARS = [
    "mu_omega",
    "sigma_omega",
    "sigma_direct",
    "sigma_collateral"
]

SUMMARY_VARS = [
    "mu_omega",
    "sigma_omega",
    "sigma_direct",
    "sigma_collateral",
    "Z"
]


def make_output_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    TRACEPLOT_DIR.mkdir(parents=True, exist_ok=True)
    SENSITIVITY_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def clean_label(value):
    return str(value).replace(".", "p").replace(" ", "_").replace("Â²", "2")


def scenario_tag(z_prior_name, mu_prior_sd, sigma_prior_sd):
    return f"{clean_label(z_prior_name)}_mu{mu_prior_sd}_sigma{sigma_prior_sd}"



def check_required_columns(df, required_cols, dataset_name):
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def load_and_subsample_data():
    direct = pd.read_csv(DIRECT_DATA)
    collateral = pd.read_csv(COLLATERAL_DATA)

    check_required_columns(
        direct,
        DIRECT_COLS,
        "Direct dataset"
    )

    check_required_columns(
        collateral,
        COLLATERAL_COLS,
        "Collateral dataset"
    )

    n_direct = min(N_DIRECT, len(direct))
    n_collateral = min(N_COLLATERAL, len(collateral))

    direct = (
        direct
        .sample(n=n_direct, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    collateral = (
        collateral
        .sample(n=n_collateral, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    Y = direct[DIRECT_COLS].to_numpy(dtype=float)
    C = collateral[COLLATERAL_COLS].to_numpy(dtype=float)

    return Y, C


def flatten_observed_values(X):

    values = X[np.isfinite(X)].astype(float)

    if values.size == 0:
        raise ValueError("No observed values found after removing missing values.")

    return values


def make_sigma_prior(spec):
    if spec["dist"] == "HalfCauchy":
        return pm.HalfCauchy(
            "sigma_omega",
            **spec["kwargs"]
        )

    if spec["dist"] == "HalfNormal":
        return pm.HalfNormal(
            "sigma_omega",
            **spec["kwargs"]
        )

    raise ValueError(f"Unsupported sigma prior specification: {spec}")



def fit_one_scenario(
    Y_obs,
    C_obs,
    mu_prior_sd,
    sigma_omega_prior_spec,
    z_prior_spec
):
    with pm.Model() as model:


        mu_omega = pm.Normal(
            "mu_omega",
            mu=0.0,
            sigma=mu_prior_sd
        )

        sigma_omega = make_sigma_prior(
            sigma_omega_prior_spec
        )

        sigma_direct = pm.TruncatedNormal(
            "sigma_direct",
            mu=1.20,
            sigma=0.30,
            lower=0.0
        )

        sigma_collateral = pm.TruncatedNormal(
            "sigma_collateral",
            mu=0.80,
            sigma=0.25,
            lower=0.0
        )


        Z_raw = pm.Beta(
            "Z_raw",
            alpha=z_prior_spec["alpha"],
            beta=z_prior_spec["beta"]
        )

        Z = pm.Deterministic(
            "Z",
            Z_LOWER + (Z_UPPER - Z_LOWER) * Z_raw
        )

        # Independent direct and collateral likelihood
        

        sd_direct = pm.math.sqrt(
            sigma_omega**2 + sigma_direct**2 + 1e-6
        )

        sd_collateral = pm.math.sqrt(
            sigma_omega**2 + sigma_collateral**2 + 1e-6
        )


        y_dist = pm.Normal.dist(
            mu=mu_omega,
            sigma=sd_direct
        )

        y_logp = pm.logp(
            y_dist,
            Y_obs
        )

        pm.Potential(
            "weighted_direct_loglik",
            Z * pm.math.sum(y_logp)
        )

        c_dist = pm.Normal.dist(
            mu=mu_omega,
            sigma=sd_collateral
        )

        c_logp = pm.logp(
            c_dist,
            C_obs
        )

        pm.Potential(
            "weighted_collateral_loglik",
            (1.0 - Z) * pm.math.sum(c_logp)
        )


        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            progressbar=True
        )

    return trace



def summarize_scenario(
    trace,
    mu_prior_sd,
    sigma_omega_prior_name,
    z_prior_name,
    elapsed_seconds
):
    summary = az.summary(
        trace,
        var_names=SUMMARY_VARS
    )

    n_divergences = int(
        np.asarray(trace.sample_stats["diverging"]).sum()
    )

    rows = []


    for param in PARAMETER_VARS:

        true_val = TRUE_VALUES[param]

        draws = (
            trace
            .posterior[param]
            .values
            .reshape(-1)
        )

        mean_est = float(np.mean(draws))

        lower = float(np.quantile(draws, 0.025))
        upper = float(np.quantile(draws, 0.975))

        bias = mean_est - true_val
        mse = bias**2
        rmse = np.sqrt(mse)

        coverage = lower <= true_val <= upper

        ess = (
            float(summary.loc[param, "ess_bulk"])
            if "ess_bulk" in summary.columns
            else np.nan
        )

        rhat = (
            float(summary.loc[param, "r_hat"])
            if "r_hat" in summary.columns
            else (
                float(summary.loc[param, "rhat"])
                if "rhat" in summary.columns
                else np.nan
            )
        )

        rows.append({

            "Structure": "Independent",

            "ZPrior": z_prior_name,

            "MuPrior": f"N(0,{mu_prior_sd}²)",

            "SigmaPrior": sigma_omega_prior_name,

            "Parameter": param,

            "True": true_val,

            "PosteriorMean": mean_est,

            "Bias": bias,

            "MSE": mse,

            "RMSE": rmse,

            "Lower95CrI": lower,

            "Upper95CrI": upper,

            "Coverage": coverage,

            "ESS_Bulk": ess,

            "Rhat": rhat,

            "Divergences": n_divergences,

            "ElapsedSeconds": elapsed_seconds
        })


    z_draws = (
        trace
        .posterior["Z"]
        .values
        .reshape(-1)
    )

    z_mean = float(np.mean(z_draws))
    z_lower = float(np.quantile(z_draws, 0.025))
    z_upper = float(np.quantile(z_draws, 0.975))

    z_ess = (
        float(summary.loc["Z", "ess_bulk"])
        if "ess_bulk" in summary.columns
        else np.nan
    )

    z_rhat = (
        float(summary.loc["Z", "r_hat"])
        if "r_hat" in summary.columns
        else (
            float(summary.loc["Z", "rhat"])
            if "rhat" in summary.columns
            else np.nan
        )
    )

    rows.append({

        "Structure": "Independent",

        "ZPrior": z_prior_name,

        "MuPrior": f"N(0,{mu_prior_sd}²)",

        "SigmaPrior": sigma_omega_prior_name,

        "Parameter": "Z",

        "True": np.nan,

        "PosteriorMean": z_mean,

        "Bias": np.nan,

        "MSE": np.nan,

        "RMSE": np.nan,

        "Lower95CrI": z_lower,

        "Upper95CrI": z_upper,

        "Coverage": np.nan,

        "ESS_Bulk": z_ess,

        "Rhat": z_rhat,

        "Divergences": n_divergences,

        "ElapsedSeconds": elapsed_seconds
    })

    return pd.DataFrame(rows)


def save_trace_file(trace, trace_file):
    try:
        trace.to_netcdf(trace_file)
        return trace_file, ""
    except Exception as exc:
        pickle_file = trace_file.with_suffix(".pkl")

        try:
            with open(pickle_file, "wb") as file:
                pickle.dump(trace, file)

            return pickle_file, ""
        except Exception as pickle_exc:
            return None, (
                "Trace file was not saved. "
                f"NetCDF error: {exc}. "
                f"Pickle fallback error: {pickle_exc}"
            )


def save_traceplot(trace, z_prior_name, mu_prior_sd, sigma_omega_prior_spec, scenario_number):
    tag = scenario_tag(z_prior_name, mu_prior_sd, sigma_omega_prior_spec)

    trace_file = TRACE_DIR / f"scenario_{scenario_number}_{tag}.nc"

    trace_file, trace_save_warning = save_trace_file(
        trace,
        trace_file
    )

    az.plot_trace(
        trace,
        var_names=SUMMARY_VARS,
        backend="matplotlib"
    )

    plt.tight_layout()

    plot_file = TRACEPLOT_DIR / f"traceplot_scenario_{scenario_number}_{tag}.png"

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return trace_file, plot_file, trace_save_warning



def create_parameter_comparison_tables(final_df):
    parameter_df = final_df[
        final_df["Parameter"].isin(PARAMETER_VARS)
    ].copy()

    z_summary_df = final_df[
        final_df["Parameter"] == "Z"
    ].copy()

    ranking_df = (
        parameter_df
        .groupby(
            ["ZPrior", "MuPrior", "SigmaPrior"],
            as_index=False
        )
        .agg(
            AvgRMSE=("RMSE", "mean"),
            CoverageRate=("Coverage", "mean"),
            MinESS=("ESS_Bulk", "min"),
            MaxRhat=("Rhat", "max"),
            Divergences=("Divergences", "max"),
            ElapsedSeconds=("ElapsedSeconds", "max")
        )
        .sort_values(
            ["AvgRMSE", "MaxRhat", "MinESS"],
            ascending=[True, True, False]
        )
    )

    parameter_summary_df = (
        parameter_df
        .groupby(
            ["Parameter"],
            as_index=False
        )
        .agg(
            MeanPosterior=("PosteriorMean", "mean"),
            MeanRMSE=("RMSE", "mean"),
            MinRMSE=("RMSE", "min"),
            MaxRMSE=("RMSE", "max"),
            CoverageRate=("Coverage", "mean"),
            MinESS=("ESS_Bulk", "min"),
            MaxRhat=("Rhat", "max"),
            MaxDivergences=("Divergences", "max")
        )
        .sort_values("MeanRMSE")
    )

    z_prior_summary_df = (
        parameter_df
        .groupby(
            ["ZPrior", "Parameter"],
            as_index=False
        )
        .agg(
            MeanPosterior=("PosteriorMean", "mean"),
            MeanRMSE=("RMSE", "mean"),
            CoverageRate=("Coverage", "mean"),
            MinESS=("ESS_Bulk", "min"),
            MaxRhat=("Rhat", "max")
        )
        .sort_values(
            ["Parameter", "ZPrior"]
        )
    )

    sigma_omega_summary_df = (
        parameter_df
        .groupby(
            ["SigmaPrior", "Parameter"],
            as_index=False
        )
        .agg(
            MeanPosterior=("PosteriorMean", "mean"),
            MeanRMSE=("RMSE", "mean"),
            CoverageRate=("Coverage", "mean"),
            MinESS=("ESS_Bulk", "min"),
            MaxRhat=("Rhat", "max")
        )
        .sort_values(
            ["Parameter", "SigmaPrior"]
        )
    )

    learned_z_summary_df = (
        z_summary_df
        .groupby(
            ["ZPrior"],
            as_index=False
        )
        .agg(
            MeanZ=("PosteriorMean", "mean"),
            MinZ=("PosteriorMean", "min"),
            MaxZ=("PosteriorMean", "max"),
            MeanLower95CrI=("Lower95CrI", "mean"),
            MeanUpper95CrI=("Upper95CrI", "mean"),
            MinESS=("ESS_Bulk", "min"),
            MaxRhat=("Rhat", "max")
        )
    )

    best_by_parameter_df = (
        parameter_df
        .sort_values(
            ["Parameter", "RMSE", "Rhat", "ESS_Bulk"],
            ascending=[True, True, True, False]
        )
        .groupby(
            "Parameter",
            as_index=False
        )
        .first()
    )

    pivot_mean_df = parameter_df.pivot_table(
        index=["ZPrior", "MuPrior", "SigmaPrior"],
        columns="Parameter",
        values="PosteriorMean",
        aggfunc="first"
    ).reset_index()

    pivot_mean_df.columns.name = None

    pivot_rmse_df = parameter_df.pivot_table(
        index=["ZPrior", "MuPrior", "SigmaPrior"],
        columns="Parameter",
        values="RMSE",
        aggfunc="first"
    ).reset_index()

    pivot_rmse_df.columns.name = None

    ranking_df.to_csv(TABLE_DIR / "scenario_ranking_by_avg_rmse.csv", index=False)
    parameter_summary_df.to_csv(TABLE_DIR / "parameter_summary.csv", index=False)
    z_prior_summary_df.to_csv(TABLE_DIR / "z_prior_sensitivity_summary.csv", index=False)
    sigma_omega_summary_df.to_csv(TABLE_DIR / "sigma_omega_prior_sensitivity_summary.csv", index=False)
    learned_z_summary_df.to_csv(TABLE_DIR / "learned_z_summary.csv", index=False)
    best_by_parameter_df.to_csv(TABLE_DIR / "best_scenario_by_parameter.csv", index=False)
    pivot_mean_df.to_csv(TABLE_DIR / "posterior_mean_comparison_wide.csv", index=False)
    pivot_rmse_df.to_csv(TABLE_DIR / "rmse_comparison_wide.csv", index=False)

    excel_path = TABLE_DIR / "Independent_sampledZ_parameter_comparison_tables.xlsx"

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            ranking_df.to_excel(writer, sheet_name="Scenario ranking", index=False)
            parameter_summary_df.to_excel(writer, sheet_name="Parameter summary", index=False)
            z_prior_summary_df.to_excel(writer, sheet_name="Z prior sensitivity", index=False)
            sigma_omega_summary_df.to_excel(writer, sheet_name="SigmaOmega prior sensitivity", index=False)
            learned_z_summary_df.to_excel(writer, sheet_name="Learned Z", index=False)
            best_by_parameter_df.to_excel(writer, sheet_name="Best by parameter", index=False)
            pivot_mean_df.to_excel(writer, sheet_name="Posterior means wide", index=False)
            pivot_rmse_df.to_excel(writer, sheet_name="RMSE wide", index=False)
    except ImportError:
        print("\nopenpyxl is not installed, so the Excel workbook was skipped.")
        print("CSV comparison tables were still saved.")

    return {
        "ranking": ranking_df,
        "parameter_summary": parameter_summary_df,
        "z_prior_summary": z_prior_summary_df,
        "sigma_omega_summary": sigma_omega_summary_df,
        "learned_z_summary": learned_z_summary_df,
        "best_by_parameter": best_by_parameter_df,
        "posterior_mean_wide": pivot_mean_df,
        "rmse_wide": pivot_rmse_df,
        "z_rows": z_summary_df
    }

# SENSITIVITY PLOTS

def save_current_figure(path):
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_scenario_ranking(ranking_df):
    top_df = ranking_df.head(15).copy()

    top_df["Scenario"] = (
        top_df["ZPrior"].astype(str)
        + " | " + top_df["MuPrior"].astype(str)
        + " | " + top_df["SigmaPrior"].astype(str)
    )

    top_df = top_df.sort_values("AvgRMSE", ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(top_df["Scenario"], top_df["AvgRMSE"])
    plt.xlabel("Average RMSE")
    plt.ylabel("Scenario")
    plt.title("Top 15 independent sampled-Z scenarios by average RMSE")
    plt.grid(axis="x", alpha=0.3)

    save_current_figure(
        SENSITIVITY_PLOT_DIR / "top15_scenarios_by_avg_rmse.png"
    )


def plot_coverage_by_parameter(parameter_summary_df):
    plt.figure(figsize=(9, 5))
    plt.bar(
        parameter_summary_df["Parameter"],
        parameter_summary_df["CoverageRate"]
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Coverage rate")
    plt.xlabel("Parameter")
    plt.title("Coverage rate by parameter")
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(
        SENSITIVITY_PLOT_DIR / "coverage_rate_by_parameter.png"
    )


def plot_mean_rmse_by_parameter(parameter_summary_df):
    plt.figure(figsize=(9, 5))
    plt.bar(
        parameter_summary_df["Parameter"],
        parameter_summary_df["MeanRMSE"]
    )
    plt.ylabel("Mean RMSE")
    plt.xlabel("Parameter")
    plt.title("Mean RMSE by parameter")
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(
        SENSITIVITY_PLOT_DIR / "mean_rmse_by_parameter.png"
    )


def plot_z_prior_sensitivity_by_parameter(z_prior_summary_df):
    for parameter in PARAMETER_VARS:

        sub = z_prior_summary_df[
            z_prior_summary_df["Parameter"] == parameter
        ].copy()

        sub = sub.sort_values("MeanRMSE", ascending=True)

        plt.figure(figsize=(9, 5))
        plt.bar(
            sub["ZPrior"],
            sub["MeanRMSE"]
        )
        plt.ylabel("Mean RMSE")
        plt.xlabel("Z prior")
        plt.title(f"Z-prior sensitivity for {parameter}")
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", alpha=0.3)

        save_current_figure(
            SENSITIVITY_PLOT_DIR / f"z_prior_sensitivity_{parameter}.png"
        )


def plot_sigma_omega_sensitivity_by_parameter(sigma_omega_summary_df):
    for parameter in PARAMETER_VARS:

        sub = sigma_omega_summary_df[
            sigma_omega_summary_df["Parameter"] == parameter
        ].copy()

        sub = sub.sort_values("MeanRMSE", ascending=True)

        plt.figure(figsize=(10, 5))
        plt.bar(
            sub["SigmaPrior"],
            sub["MeanRMSE"]
        )
        plt.ylabel("Mean RMSE")
        plt.xlabel("SigmaOmega prior")
        plt.title(f"SigmaOmega-prior sensitivity for {parameter}")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)

        save_current_figure(
            SENSITIVITY_PLOT_DIR / f"sigma_omega_prior_sensitivity_{parameter}.png"
        )


def plot_posterior_mean_vs_true_by_parameter(parameter_df):
    true_col = "True" if "True" in parameter_df.columns else "TRUE"

    if true_col not in parameter_df.columns:
        print("True-value column not available. Skipping posterior mean plots.")
        return

    for parameter in PARAMETER_VARS:

        sub = parameter_df[
            parameter_df["Parameter"] == parameter
        ].copy()

        sub = (
            sub
            .groupby(["ZPrior"], as_index=False)
            .agg(
                MeanPosterior=("PosteriorMean", "mean"),
                TrueValue=(true_col, "mean")
            )
        )

        plt.figure(figsize=(9, 5))
        plt.bar(
            sub["ZPrior"],
            sub["MeanPosterior"],
            label="Posterior mean"
        )
        plt.axhline(
            y=sub["TrueValue"].iloc[0],
            linestyle="--",
            label="True value"
        )
        plt.ylabel("Estimate")
        plt.xlabel("Z prior")
        plt.title(f"Posterior mean versus true value: {parameter}")
        plt.xticks(rotation=20, ha="right")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        save_current_figure(
            SENSITIVITY_PLOT_DIR / f"posterior_mean_vs_true_{parameter}.png"
        )


def plot_learned_z_by_prior(z_summary_df):
    plot_df = (
        z_summary_df
        .groupby(["ZPrior"], as_index=False)
        .agg(
            MeanZ=("PosteriorMean", "mean"),
            Lower95CrI=("Lower95CrI", "mean"),
            Upper95CrI=("Upper95CrI", "mean")
        )
    )

    lower_error = plot_df["MeanZ"] - plot_df["Lower95CrI"]
    upper_error = plot_df["Upper95CrI"] - plot_df["MeanZ"]

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        plot_df["ZPrior"],
        plot_df["MeanZ"],
        yerr=[lower_error, upper_error],
        fmt="o",
        capsize=5
    )
    plt.axhline(Z_LOWER, linestyle=":", linewidth=1)
    plt.axhline(Z_UPPER, linestyle=":", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("Posterior Z")
    plt.xlabel("Z prior")
    plt.title("Learned sampled Z by prior")
    plt.grid(axis="y", alpha=0.3)

    save_current_figure(
        SENSITIVITY_PLOT_DIR / "learned_z_by_prior.png"
    )


def plot_diagnostics_scatter(ranking_df):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        ranking_df["MaxRhat"],
        ranking_df["MinESS"]
    )
    plt.axvline(1.01, linestyle="--", linewidth=1)
    plt.axvline(1.02, linestyle=":", linewidth=1)
    plt.axhline(100, linestyle="--", linewidth=1)
    plt.xlabel("Maximum Rhat across parameters")
    plt.ylabel("Minimum ESS across parameters")
    plt.title("Scenario diagnostic plot: ESS versus Rhat")
    plt.grid(alpha=0.3)

    save_current_figure(
        SENSITIVITY_PLOT_DIR / "diagnostic_min_ess_vs_max_rhat.png"
    )


def generate_sensitivity_plots(final_df, tables):
    parameter_df = final_df[
        final_df["Parameter"].isin(PARAMETER_VARS)
    ].copy()

    plot_scenario_ranking(tables["ranking"])
    plot_coverage_by_parameter(tables["parameter_summary"])
    plot_mean_rmse_by_parameter(tables["parameter_summary"])
    plot_z_prior_sensitivity_by_parameter(tables["z_prior_summary"])
    plot_sigma_omega_sensitivity_by_parameter(tables["sigma_omega_summary"])
    plot_posterior_mean_vs_true_by_parameter(parameter_df)
    plot_learned_z_by_prior(tables["z_rows"])
    plot_diagnostics_scatter(tables["ranking"])


def main():

    make_output_dirs()

    print("\n================================================")
    print("INDEPENDENT STRUCTURE SAMPLED-Z SENSITIVITY ANALYSIS")
    print("Marginal independent likelihood; no latent-state sampling")
    print("Z sampled from Beta and constrained deterministically")
    print("================================================")
    print("PyMC version:", pm.__version__)
    print("ArviZ version:", az.__version__)
    print("Output directory:", OUTDIR)
    print("Draws:", DRAWS)
    print("Tune:", TUNE)
    print("Chains:", CHAINS)
    print("Cores:", CORES)
    print("Target accept:", TARGET_ACCEPT)
    print("Z constraint:", f"{Z_LOWER} < Z < {Z_UPPER}")

    Y, C = load_and_subsample_data()

    Y_obs = flatten_observed_values(Y)
    C_obs = flatten_observed_values(C)

    print("\nDirect matrix shape:", Y.shape)
    print("Collateral matrix shape:", C.shape)
    print("Observed direct values:", len(Y_obs))
    print("Observed collateral values:", len(C_obs))

    all_results = []
    diagnostics = []

    scenario_number = 0

    for z_prior_spec in Z_PRIOR_SPECS:

        for mu_prior_sd in MU_PRIOR_SDS:

            for sigma_omega_prior_spec in SIGMA_PRIOR_SPECS:

                scenario_number += 1

                z_prior_name = z_prior_spec["name"]
                sigma_omega_name = sigma_omega_prior_spec["name"]

                print("\n" + "=" * 80)
                print(
                    f"Scenario {scenario_number}: "
                    f"ZPrior={z_prior_name}, "
                    f"MuPrior=N(0,{mu_prior_sd}²), "
                    f"SigmaOmegaPrior={sigma_omega_name}"
                )
                print("=" * 80)

                start_time = time.time()

                try:

                    trace = fit_one_scenario(
                        Y_obs=Y_obs,
                        C_obs=C_obs,
                        mu_prior_sd=mu_prior_sd,
                        sigma_omega_prior_spec=sigma_omega_prior_spec,
                        z_prior_spec=z_prior_spec
                    )

                    elapsed = time.time() - start_time

                    scenario_df = summarize_scenario(
                        trace=trace,
                        mu_prior_sd=mu_prior_sd,
                        sigma_omega_prior_name=sigma_omega_name,
                        z_prior_name=z_prior_name,
                        elapsed_seconds=elapsed
                    )

                    all_results.append(scenario_df)

                    trace_file, traceplot_file, trace_save_warning = save_traceplot(
                        trace=trace,
                        z_prior_name=z_prior_name,
                        mu_prior_sd=mu_prior_sd,
                        sigma_omega_prior_spec=sigma_omega_name,
                        scenario_number=scenario_number
                    )

                    n_div = int(
                        scenario_df["Divergences"].iloc[0]
                    )

                    diagnostics.append({
                        "Scenario": scenario_number,
                        "ZPrior": z_prior_name,
                        "MuPriorSD": mu_prior_sd,
                        "SigmaOmegaPrior": sigma_omega_name,
                        "Status": "success",
                        "Divergences": n_div,
                        "ElapsedSeconds": elapsed,
                        "TraceFile": str(trace_file) if trace_file else "",
                        "TraceplotFile": str(traceplot_file),
                        "TraceSaveWarning": trace_save_warning,
                        "Error": ""
                    })

                    if trace_save_warning:
                        print("\nTrace file warning:")
                        print(trace_save_warning)

                    scenario_path = OUTDIR / (
                        f"scenario_{scenario_number}_{z_prior_name}_mu{mu_prior_sd}_{sigma_omega_name}.csv"
                    )

                    scenario_df.to_csv(
                        scenario_path,
                        index=False
                    )

                    progress_df = pd.concat(
                        all_results,
                        ignore_index=True
                    )

                    progress_path = OUTDIR / "Independent_sampledZ_progress.csv"

                    progress_df.to_csv(
                        progress_path,
                        index=False
                    )

                    print("\nScenario results:")
                    print(
                        scenario_df[
                            [
                                "Parameter",
                                "PosteriorMean",
                                "RMSE",
                                "Coverage",
                                "ESS_Bulk",
                                "Rhat",
                                "Divergences",
                                "ElapsedSeconds"
                            ]
                        ].round(4).to_string(index=False)
                    )

                except Exception as exc:

                    elapsed = time.time() - start_time

                    diagnostics.append({
                        "Scenario": scenario_number,
                        "ZPrior": z_prior_name,
                        "MuPriorSD": mu_prior_sd,
                        "SigmaOmegaPrior": sigma_omega_name,
                        "Status": "failed",
                        "Divergences": np.nan,
                        "ElapsedSeconds": elapsed,
                        "TraceFile": "",
                        "TraceplotFile": "",
                        "TraceSaveWarning": "",
                        "Error": str(exc)
                    })

                    print("\nScenario failed.")
                    print("Error:", exc)

                diagnostics_df = pd.DataFrame(diagnostics)

                diagnostics_path = OUTDIR / "Independent_sampledZ_diagnostics.csv"

                diagnostics_df.to_csv(
                    diagnostics_path,
                    index=False
                )


    if all_results:

        final_df = pd.concat(
            all_results,
            ignore_index=True
        )

        final_path = OUTDIR / "Independent_sampledZ_final_results.csv"

        final_df.to_csv(
            final_path,
            index=False
        )

        tables = create_parameter_comparison_tables(final_df)
        generate_sensitivity_plots(final_df, tables)

        ranking_df = tables["ranking"]

        ranking_path = OUTDIR / "Independent_sampledZ_scenario_ranking.csv"

        ranking_df.to_csv(
            ranking_path,
            index=False
        )

        z_summary_df = tables["z_rows"]

        z_summary_path = OUTDIR / "Independent_sampledZ_Z_summary.csv"

        z_summary_df.to_csv(
            z_summary_path,
            index=False
        )

        print("\n" + "=" * 80)
        print("FINAL SCENARIO RANKING")
        print("=" * 80)

        print(
            ranking_df
            .round(4)
            .to_string(index=False)
        )

        print("\n" + "=" * 80)
        print("POSTERIOR Z SUMMARY")
        print("=" * 80)

        print(
            z_summary_df[
                [
                    "ZPrior",
                    "MuPrior",
                    "SigmaPrior",
                    "PosteriorMean",
                    "Lower95CrI",
                    "Upper95CrI",
                    "ESS_Bulk",
                    "Rhat"
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

        print("\nSaved files:")
        print(final_path)
        print(ranking_path)
        print(z_summary_path)
        print(OUTDIR / "Independent_sampledZ_progress.csv")
        print(OUTDIR / "Independent_sampledZ_diagnostics.csv")
        print(TRACE_DIR)
        print(TRACEPLOT_DIR)
        print(SENSITIVITY_PLOT_DIR)
        print(TABLE_DIR)

    else:

        print("\nNo successful scenarios completed.")

    print("\nDONE.")


if __name__ == "__main__":
    main()
