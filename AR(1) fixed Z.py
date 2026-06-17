# AR(1) FIXED-Z SENSITIVITY ANALYSIS


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
    r"F:\gacheri's documents\research NCDs\BIBTEX\New folder\Simulated Datasets\Main Sensitivity\AR1_fixedZ_results"
)

TRACE_DIR = OUTDIR / "posterior_traces"
TRACEPLOT_DIR = OUTDIR / "traceplots"
SENSITIVITY_PLOT_DIR = OUTDIR / "sensitivity_plots"
TABLE_DIR = OUTDIR / "comparison_tables"

DIRECT_COLS = ["Y_t1", "Y_t2", "Y_t3"]
COLLATERAL_COLS = ["C_t1", "C_t2", "C_t3"]

# Reduced sample size for short sensitivity runs
N_DIRECT = 150
N_COLLATERAL = 200

# Fixed credibility weights
Z_VALUES = [0.9, 0.7, 0.5]

# Latent mean prior sensitivity
MU_PRIOR_SDS = [2, 5, 10]

# Tau prior sensitivity and robustness check
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

# MCMC
DRAWS = 1000
TUNE = 1500
CHAINS = 4
CORES = 1

TARGET_ACCEPT = 0.95
RANDOM_SEED = 123

TRUE_VALUES = {
    "mu_omega": 0.50,
    "sigma_omega": 1.00,
    "rho": 0.85,
    "sigma_direct": 1.20,
    "sigma_collateral": 0.80
}

MAIN_VARS = [
    "mu_omega",
    "sigma_omega",
    "rho",
    "sigma_direct",
    "sigma_collateral"
]


def make_output_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    TRACEPLOT_DIR.mkdir(parents=True, exist_ok=True)
    SENSITIVITY_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def clean_label(value):
    return str(value).replace(".", "p").replace(" ", "_").replace("Â²", "2")


def scenario_tag(z_value, mu_prior_sd, sigma_prior_name):
    return f"Z{clean_label(z_value)}_mu{mu_prior_sd}_{sigma_prior_name}"



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


def group_rows_by_missing_pattern(X):

    groups = {}

    for row in X:

        mask = tuple(np.isfinite(row))

        if not any(mask):
            continue

        observed_values = row[np.isfinite(row)]

        if mask not in groups:
            groups[mask] = []

        groups[mask].append(observed_values)

    grouped = {}

    for mask, rows in groups.items():
        grouped[mask] = np.asarray(rows, dtype=float)

    return grouped


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

    raise ValueError(f"Unsupported tau prior specification: {spec}")


def ar1_mean_vector(mu_omega, rho):

    return pt.stack([
        mu_omega,
        rho * mu_omega,
        rho**2 * mu_omega
    ])


def ar1_latent_covariance(sigma, rho):
    tau = sigma
    rho2 = rho**2
    rho3 = rho**3
    rho4 = rho**4

    Sigma = tau**2 * pt.stack([
        pt.stack([1.0, rho, rho2]),
        pt.stack([rho, 1.0 + rho2, rho + rho3]),
        pt.stack([rho2, rho + rho3, 1.0 + rho2 + rho4])
    ])

    return Sigma



def fit_one_scenario(
    Y_groups,
    C_groups,
    z_value,
    mu_prior_sd,
    sigma_prior_spec
):
    with pm.Model() as model:


        mu_omega = pm.Normal(
            "mu_omega",
            mu=0.0,
            sigma=mu_prior_sd
        )

        sigma_omega = make_sigma_prior(
            sigma_prior_spec
        )

        rho = pm.Beta(
            "rho",
            alpha=8.0,
            beta=2.0
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

        mu_vec = ar1_mean_vector(
            mu_omega=mu_omega,
            rho=rho
        )

        Sigma_latent = ar1_latent_covariance(
            sigma=sigma_omega,
            rho=rho
        )



        Sigma_direct = (
            Sigma_latent
            +
            pt.eye(3) * sigma_direct**2
            +
            pt.eye(3) * 1e-6
        )

        Sigma_collateral = (
            Sigma_latent
            +
            pt.eye(3) * sigma_collateral**2
            +
            pt.eye(3) * 1e-6
        )

        # Weighted direct likelihood
      

        for mask, block in Y_groups.items():

            idx = np.where(
                np.asarray(mask)
            )[0].astype("int64")

            label = "".join(
                ["1" if val else "0" for val in mask]
            )

            y_dist = pm.MvNormal.dist(
                mu=mu_vec[idx],
                cov=Sigma_direct[idx, :][:, idx]
            )

            y_logp = pm.logp(
                y_dist,
                block
            )

            pm.Potential(
                f"weighted_Y_loglik_{label}",
                z_value * pt.sum(y_logp)
            )

    
        # Weighted collateral likelihood


        for mask, block in C_groups.items():

            idx = np.where(
                np.asarray(mask)
            )[0].astype("int64")

            label = "".join(
                ["1" if val else "0" for val in mask]
            )

            c_dist = pm.MvNormal.dist(
                mu=mu_vec[idx],
                cov=Sigma_collateral[idx, :][:, idx]
            )

            c_logp = pm.logp(
                c_dist,
                block
            )

            pm.Potential(
                f"weighted_C_loglik_{label}",
                (1.0 - z_value) * pt.sum(c_logp)
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
    z_value,
    mu_prior_sd,
    sigma_prior_name,
    elapsed_seconds
):
    summary = az.summary(
        trace,
        var_names=MAIN_VARS
    )

    n_divergences = int(
        np.asarray(trace.sample_stats["diverging"]).sum()
    )

    rows = []

    for param in MAIN_VARS:

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
            "Structure": "AR1",
            "Z": z_value,
            "MuPrior": f"N(0,{mu_prior_sd}Â²)",
            "SigmaPrior": sigma_prior_name,
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

    rows.append({
        "Structure": "AR1",
        "Z": z_value,
        "MuPrior": f"N(0,{mu_prior_sd}Â²)",
        "SigmaPrior": sigma_prior_name,
        "Parameter": "Z_fixed",
        "True": np.nan,
        "PosteriorMean": z_value,
        "Bias": np.nan,
        "MSE": np.nan,
        "RMSE": np.nan,
        "Lower95CrI": np.nan,
        "Upper95CrI": np.nan,
        "Coverage": np.nan,
        "ESS_Bulk": np.nan,
        "Rhat": np.nan,
        "Divergences": n_divergences,
        "ElapsedSeconds": elapsed_seconds
    })

    return pd.DataFrame(rows)


# TRACEPLOTS


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


def save_traceplot(trace, z_value, mu_prior_sd, sigma_prior_name, scenario_number):
    tag = scenario_tag(z_value, mu_prior_sd, sigma_prior_name)

    trace_file = TRACE_DIR / f"scenario_{scenario_number}_{tag}.nc"

    trace_file, trace_save_warning = save_trace_file(
        trace,
        trace_file
    )

    az.plot_trace(
        trace,
        var_names=MAIN_VARS,
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

# PARAMETER COMPARISON TABLES

def create_parameter_comparison_tables(final_df):
    parameter_df = final_df[
        final_df["Parameter"].isin(MAIN_VARS)
    ].copy()

    ranking_df = (
        parameter_df
        .groupby(
            ["Z", "MuPrior", "SigmaPrior"],
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

    z_summary_df = (
        parameter_df
        .groupby(
            ["Z", "Parameter"],
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
            ["Parameter", "Z"]
        )
    )

    sigma_summary_df = (
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
        index=["Z", "MuPrior", "SigmaPrior"],
        columns="Parameter",
        values="PosteriorMean",
        aggfunc="first"
    ).reset_index()

    pivot_mean_df.columns.name = None

    pivot_rmse_df = parameter_df.pivot_table(
        index=["Z", "MuPrior", "SigmaPrior"],
        columns="Parameter",
        values="RMSE",
        aggfunc="first"
    ).reset_index()

    pivot_rmse_df.columns.name = None

    ranking_df.to_csv(TABLE_DIR / "scenario_ranking_by_avg_rmse.csv", index=False)
    parameter_summary_df.to_csv(TABLE_DIR / "parameter_summary.csv", index=False)
    z_summary_df.to_csv(TABLE_DIR / "z_sensitivity_summary.csv", index=False)
    sigma_summary_df.to_csv(TABLE_DIR / "sigma_prior_sensitivity_summary.csv", index=False)
    best_by_parameter_df.to_csv(TABLE_DIR / "best_scenario_by_parameter.csv", index=False)
    pivot_mean_df.to_csv(TABLE_DIR / "posterior_mean_comparison_wide.csv", index=False)
    pivot_rmse_df.to_csv(TABLE_DIR / "rmse_comparison_wide.csv", index=False)

    excel_path = TABLE_DIR / "AR1_fixedZ_parameter_comparison_tables.xlsx"

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            ranking_df.to_excel(writer, sheet_name="Scenario ranking", index=False)
            parameter_summary_df.to_excel(writer, sheet_name="Parameter summary", index=False)
            z_summary_df.to_excel(writer, sheet_name="Z sensitivity", index=False)
            sigma_summary_df.to_excel(writer, sheet_name="Sigma sensitivity", index=False)
            best_by_parameter_df.to_excel(writer, sheet_name="Best by parameter", index=False)
            pivot_mean_df.to_excel(writer, sheet_name="Posterior means wide", index=False)
            pivot_rmse_df.to_excel(writer, sheet_name="RMSE wide", index=False)
    except ImportError:
        print("\nopenpyxl is not installed, so the Excel workbook was skipped.")
        print("CSV comparison tables were still saved.")

    return {
        "ranking": ranking_df,
        "parameter_summary": parameter_summary_df,
        "z_summary": z_summary_df,
        "sigma_summary": sigma_summary_df,
        "best_by_parameter": best_by_parameter_df,
        "posterior_mean_wide": pivot_mean_df,
        "rmse_wide": pivot_rmse_df
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
        "Z=" + top_df["Z"].astype(str)
        + " | " + top_df["MuPrior"].astype(str)
        + " | " + top_df["SigmaPrior"].astype(str)
    )

    top_df = top_df.sort_values("AvgRMSE", ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(top_df["Scenario"], top_df["AvgRMSE"])
    plt.xlabel("Average RMSE")
    plt.ylabel("Scenario")
    plt.title("Top 15 AR(1) fixed-Z scenarios by average RMSE")
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


def plot_z_sensitivity_by_parameter(z_summary_df):
    for parameter in MAIN_VARS:

        sub = z_summary_df[
            z_summary_df["Parameter"] == parameter
        ].copy()

        sub = sub.sort_values("Z")

        plt.figure(figsize=(8, 5))
        plt.plot(
            sub["Z"],
            sub["MeanRMSE"],
            marker="o"
        )
        plt.xlabel("Fixed Z")
        plt.ylabel("Mean RMSE")
        plt.title(f"Z sensitivity for {parameter}: mean RMSE")
        plt.grid(alpha=0.3)

        save_current_figure(
            SENSITIVITY_PLOT_DIR / f"z_sensitivity_rmse_{parameter}.png"
        )


def plot_sigma_sensitivity_by_parameter(sigma_summary_df):
    for parameter in MAIN_VARS:

        sub = sigma_summary_df[
            sigma_summary_df["Parameter"] == parameter
        ].copy()

        sub = sub.sort_values("MeanRMSE", ascending=True)

        plt.figure(figsize=(10, 5))
        plt.bar(
            sub["SigmaPrior"],
            sub["MeanRMSE"]
        )
        plt.ylabel("Mean RMSE")
        plt.xlabel("Sigma prior")
        plt.title(f"Sigma-prior sensitivity for {parameter}")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)

        save_current_figure(
            SENSITIVITY_PLOT_DIR / f"sigma_prior_sensitivity_{parameter}.png"
        )


def plot_posterior_mean_vs_true_by_parameter(parameter_df):
    true_col = "True" if "True" in parameter_df.columns else "TRUE"

    if true_col not in parameter_df.columns:
        print("True-value column not available. Skipping posterior mean plots.")
        return

    for parameter in MAIN_VARS:

        sub = parameter_df[
            parameter_df["Parameter"] == parameter
        ].copy()

        sub = (
            sub
            .groupby(["Z"], as_index=False)
            .agg(
                MeanPosterior=("PosteriorMean", "mean"),
                TrueValue=(true_col, "mean")
            )
            .sort_values("Z")
        )

        plt.figure(figsize=(8, 5))
        plt.plot(
            sub["Z"],
            sub["MeanPosterior"],
            marker="o",
            label="Posterior mean"
        )
        plt.axhline(
            y=sub["TrueValue"].iloc[0],
            linestyle="--",
            label="True value"
        )
        plt.xlabel("Fixed Z")
        plt.ylabel("Estimate")
        plt.title(f"Posterior mean versus true value: {parameter}")
        plt.legend()
        plt.grid(alpha=0.3)

        save_current_figure(
            SENSITIVITY_PLOT_DIR / f"posterior_mean_vs_true_{parameter}.png"
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
        final_df["Parameter"].isin(MAIN_VARS)
    ].copy()

    plot_scenario_ranking(tables["ranking"])
    plot_coverage_by_parameter(tables["parameter_summary"])
    plot_mean_rmse_by_parameter(tables["parameter_summary"])
    plot_z_sensitivity_by_parameter(tables["z_summary"])
    plot_sigma_sensitivity_by_parameter(tables["sigma_summary"])
    plot_posterior_mean_vs_true_by_parameter(parameter_df)
    plot_diagnostics_scatter(tables["ranking"])


def main():

    make_output_dirs()

    print("\n================================================")
    print("AR(1) FIXED-Z SENSITIVITY ANALYSIS")
    print("Marginal AR(1) likelihood; no latent-state sampling")
    print("================================================")
    print("PyMC version:", pm.__version__)
    print("ArviZ version:", az.__version__)
    print("Output directory:", OUTDIR)
    print("Draws:", DRAWS)
    print("Tune:", TUNE)
    print("Chains:", CHAINS)
    print("Cores:", CORES)
    print("Target accept:", TARGET_ACCEPT)

    Y, C = load_and_subsample_data()

    print("\nDirect matrix shape:", Y.shape)
    print("Collateral matrix shape:", C.shape)
    print("Observed direct values:", np.isfinite(Y).sum())
    print("Observed collateral values:", np.isfinite(C).sum())

    Y_groups = group_rows_by_missing_pattern(
        Y
    )

    C_groups = group_rows_by_missing_pattern(
        C
    )

    print("\nDirect observed patterns:")
    for mask, block in Y_groups.items():
        print(mask, block.shape)

    print("\nCollateral observed patterns:")
    for mask, block in C_groups.items():
        print(mask, block.shape)

    all_results = []
    diagnostics = []
    scenario_number = 0

    for z_value in Z_VALUES:

        for mu_prior_sd in MU_PRIOR_SDS:

            for sigma_prior_spec in SIGMA_PRIOR_SPECS:

                scenario_number += 1

                sigma_name = sigma_prior_spec["name"]

                print("\n" + "=" * 80)
                print(
                    f"Scenario {scenario_number}: "
                    f"Z={z_value}, "
                    f"MuPrior=N(0,{mu_prior_sd}²), "
                    f"SigmaPrior={sigma_name}"
                )
                print("=" * 80)

                start_time = time.time()

                try:

                    trace = fit_one_scenario(
                        Y_groups=Y_groups,
                        C_groups=C_groups,
                        z_value=z_value,
                        mu_prior_sd=mu_prior_sd,
                        sigma_prior_spec=sigma_prior_spec
                    )

                    elapsed = time.time() - start_time

                    scenario_df = summarize_scenario(
                        trace=trace,
                        z_value=z_value,
                        mu_prior_sd=mu_prior_sd,
                        sigma_prior_name=sigma_name,
                        elapsed_seconds=elapsed
                    )

                    all_results.append(
                        scenario_df
                    )

                    trace_file, traceplot_file, trace_save_warning = save_traceplot(
                        trace=trace,
                        z_value=z_value,
                        mu_prior_sd=mu_prior_sd,
                        sigma_prior_name=sigma_name,
                        scenario_number=scenario_number
                    )

                    n_div = int(
                        scenario_df["Divergences"].iloc[0]
                    )

                    diagnostics.append({
                        "Scenario": scenario_number,
                        "Z": z_value,
                        "MuPriorSD": mu_prior_sd,
                        "SigmaPrior": sigma_name,
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

                    # Save scenario result
                    z_tag = str(z_value).replace(".", "p")

                    scenario_path = OUTDIR / (
                        f"scenario_Z{z_tag}_mu{mu_prior_sd}_{sigma_name}.csv"
                    )

                    scenario_df.to_csv(
                        scenario_path,
                        index=False
                    )

                    # Save progress
                    progress_df = pd.concat(
                        all_results,
                        ignore_index=True
                    )

                    progress_path = OUTDIR / "AR1_fixedZ_progress.csv"

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
                        "Z": z_value,
                        "MuPriorSD": mu_prior_sd,
                        "SigmaPrior": sigma_name,
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

                # Save diagnostics after every scenario
                diagnostics_df = pd.DataFrame(
                    diagnostics
                )

                diagnostics_path = OUTDIR / "AR1_fixedZ_diagnostics.csv"

                diagnostics_df.to_csv(
                    diagnostics_path,
                    index=False
                )

    

    if all_results:

        final_df = pd.concat(
            all_results,
            ignore_index=True
        )

        final_path = OUTDIR / "AR1_fixedZ_final_results.csv"

        final_df.to_csv(
            final_path,
            index=False
        )

        tables = create_parameter_comparison_tables(final_df)
        generate_sensitivity_plots(final_df, tables)

        ranking_df = tables["ranking"]

        ranking_path = OUTDIR / "AR1_fixedZ_scenario_ranking.csv"

        ranking_df.to_csv(
            ranking_path,
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

        print("\nSaved files:")
        print(final_path)
        print(ranking_path)
        print(OUTDIR / "AR1_fixedZ_progress.csv")
        print(OUTDIR / "AR1_fixedZ_diagnostics.csv")
        print(TRACE_DIR)
        print(TRACEPLOT_DIR)
        print(SENSITIVITY_PLOT_DIR)
        print(TABLE_DIR)

    else:

        print("\nNo successful scenarios completed.")

    print("\nDONE.")


if __name__ == "__main__":
    main()
