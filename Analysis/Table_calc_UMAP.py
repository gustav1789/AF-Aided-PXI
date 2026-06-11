import numpy as np
from umap import UMAP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2 as chi2_dist
from collections import defaultdict

from Analysis.analysis_functions import load_data

'''
This code calculates statistics over the UMAP projections and turns it into tables.
You can input in pairs how many sets of projections you want and run it over many seeds to get a correct picture.
If the proteins have weird names they can be changed by the underlying dictionary and there order is defined in lists.
'''


#---------------Correct names and define correct orders of proteins-----------------------
name_map = {
    "4KSOchainA":       "KaiB-ground",
    "KaiB_033":         "KaiB-α1",
    "KaiB_102":         "KaiB-α2",
    "KaiB_114":         "KaiB-α3",
    "5N8YchainG":       "KaiB-FS",
    "KaiB_048":         "KaiB-α4",
    "KaiB_072":         "KaiB-α5",
    "KaiB_094":         "KaiB-α6",
    "1S2H":         "Mad2-closed",
    "1DUJ":         "Mad2-open",
    "cluster_055":  "Mad2-α1",
    "cluster_081":  "Mad2-α2",
    "cluster_019":  "Mad2-α3",
    "cluster_027":  "Mad2-α4",
    "cluster_087":  "Mad2-α5",
    "RfaH-AIn":     "RfaH-AIn",
    "RfaH-α1":      "RfaH-α1",
    "RfaH-active":  "RfaH-active",
    "RfaH-α2":      "RfaH-α2",
    "RfaH-α3":      "RfaH-α3",
    "RfaH-α4":      "RfaH-α4",
    "RfaH-α5":      "RfaH-α5",
    "RfaH-α6":      "RfaH-α6",
    }

order_kaib = ["KaiB-α1", "KaiB-α2", "KaiB-α4", "KaiB-α5"]
order_mad2 = ["Mad2-α1", "Mad2-α2","Mad2-α3", "Mad2-α4"]
order_rfah = ["RfaH-α1","RfaH-α2","RfaH-α3","RfaH-α4","RfaH-α5","RfaH-α6"]
#---------------------------------------------------

def fit_pipeline(X_original, y_original, pca_components, state):
    """
    This fcn defines the "playing ground" for projection togheter with the scaler,
    defiend from the original dataset which is projected on
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)

    pca = PCA(n_components=pca_components, random_state=state)
    X_pca = pca.fit_transform(X_scaled)

    umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                metric='cosine', random_state=state)
    X_umap = umap.fit_transform(X_pca)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_umap, y_original)

    return scaler, pca, umap, knn, X_scaled, X_umap


def project_data(X_raw, scaler, pca, umap):
    """
    Project onto defiend UMAP
    """
    X_scaled = scaler.transform(X_raw)
    X_pca    = pca.transform(X_scaled)
    X_umap   = umap.transform(X_pca)
    return X_umap


def make_null(X_original_scaled, n_samples, scaler, pca, umap):
    """
    Creates a null dataset to analyse the topolpgy of UMAP-space
    """
    X_null_raw = np.random.multivariate_normal(
        X_original_scaled.mean(axis=0),
        np.cov(X_original_scaled.T),
        size=n_samples
    )
    return project_data(X_null_raw, scaler, pca, umap)


def compute_null_scale(y_null_pred, n_clusters):
    """
    In the report nut used but created as alternative way to scale the projection, where weights are defined to
    make scale the null dataset evenly on projection conformations
    """
    null_counts = np.array([np.sum(y_null_pred == i)
                            for i in range(n_clusters)], dtype=float)
    null_freq = null_counts / null_counts.sum()
    uniform   = np.ones(n_clusters) / n_clusters
    scale     = uniform / null_freq
    return null_freq, scale


def find_hcr(freq, n_top=2):     
    """
    -----Only working if there are clear HCR that are of the same size and in the right order-----
    High Confidence Regions:
      HCR-high: the n_top clusters with highest frequency
      HCR-low:  the n_top clusters with lowest frequency
    """
    sorted_idx     = np.argsort(freq)[::-1]
    hcr_high_idx   = sorted_idx[:n_top]
    hcr_low_idx    = sorted_idx[-n_top:]
    return (hcr_high_idx, freq[hcr_high_idx].sum(),
            hcr_low_idx,  freq[hcr_low_idx].sum())


def compute_distribution(y_pred, scale, null_freq, n_clusters):
    """
    Does the heavy duty work in comparing distributions against null data 
    and gives statistics. Guarding a lot against division by zero
    """
    raw_counts = np.array([np.sum(y_pred == i)
                           for i in range(n_clusters)], dtype=float)
    raw_freq   = raw_counts / raw_counts.sum()

    weighted    = raw_counts * scale
    scaled_freq = weighted / weighted.sum() if weighted.sum() > 0 else np.zeros(n_clusters)

    expected  = raw_counts.sum() * null_freq
    valid     = expected > 0
    chi2      = np.sum((raw_counts[valid] - expected[valid])**2 / expected[valid])
    dof       = valid.sum() - 1
    p_val     = chi2_dist.sf(chi2, dof)
    std_resid = (raw_counts - expected) / np.sqrt(expected + 1e-9)

    hcr_high_idx, hcr_high_total, hcr_low_idx, hcr_low_total = find_hcr(raw_freq)

    return {
        "raw_counts"    : raw_counts,
        "raw_freq"      : raw_freq,
        "scaled_freq"   : scaled_freq,
        "chi2"          : chi2,
        "p_val"         : p_val,
        "std_resid"     : std_resid,
        "hcr_high_idx"  : hcr_high_idx,
        "hcr_high_total": hcr_high_total,
        "hcr_low_idx"   : hcr_low_idx,
        "hcr_low_total" : hcr_low_total,
    }


# ── Single seed ───────────────────────────────────────────────────────────────
def run_single_seed(args):
    """
    Runs on seed
    """
    af_pkl, new_pkl, pca_components, state = args

    original_bunch = load_data(af_pkl)
    new_bunch      = load_data(new_pkl)

    X_original        = original_bunch.data
    y_original        = original_bunch.target
    target_names_orig = original_bunch.target_names
    n_clusters        = len(target_names_orig)

    y_new            = new_bunch.target
    target_names_new = new_bunch.target_names

    scaler, pca, umap, knn, X_original_scaled, _ = fit_pipeline(
        X_original, y_original, pca_components, state)

    X_new_umap  = project_data(new_bunch.data, scaler, pca, umap)
    X_null_umap = make_null(X_original_scaled, len(new_bunch.data),
                            scaler, pca, umap)

    y_new_pred  = knn.predict(X_new_umap)
    y_null_pred = knn.predict(X_null_umap)

    null_freq, scale = compute_null_scale(y_null_pred, n_clusters)

    results = {}
    for current_target in np.unique(y_new):
        pdb_name = (target_names_new[current_target]
                    if current_target < len(target_names_new)
                    else f"Kategori {current_target}")
        mask = (y_new == current_target)
        results[pdb_name] = compute_distribution(
            y_new_pred[mask], scale, null_freq, n_clusters)

    results["NULL"] = compute_distribution(y_null_pred, scale, null_freq, n_clusters)

    return results, target_names_orig


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate_results(all_results):
    """
    Computes mean and std over seeds
    """
    collected = defaultdict(lambda: defaultdict(list))

    for results, _ in all_results:
        for pdb_name, metrics in results.items():
            for key, val in metrics.items():
                collected[pdb_name][key].append(val)

    aggregated = {}
    for pdb_name, metrics in collected.items():
        aggregated[pdb_name] = {}
        for key, vals in metrics.items():
            arr = np.array(vals)
            aggregated[pdb_name][key] = {
                "mean"    : arr.mean(axis=0),
                "std"     : arr.std(axis=0),
                "per_seed": arr,           
            }
    return aggregated

#-------------For good looking prints similar to LaTex, not necessary but handy for report writing---------------------

def print_aggregated(aggregated, target_names_original, cluster_order,N_seeds):
    """
    This is AI-generated for making a really pretty print 
    """
    pdb_targets  = {k: v for k, v in aggregated.items() if k != "NULL"}
    null_metrics = aggregated.get("NULL")

    mapped_originals = [name_map.get(n, n) for n in target_names_original]
    try:
        reindex = [mapped_originals.index(name) for name in cluster_order]
    except ValueError as e:
        raise ValueError(f"cluster_order name not found in name_map output: {e}")

    ordered_names    = cluster_order
    null_rf_mean     = null_metrics["raw_freq"]["mean"]      if null_metrics is not None else None
    null_rf_per_seed = null_metrics["raw_freq"]["per_seed"]  if null_metrics is not None else None

    col_w      = 22
    struct_col = 18

    target_mapped = [name_map.get(k, k) for k in pdb_targets.keys()]
    header_cols   = target_mapped + ["Null Dataset"]
    sep           = "=" * (struct_col + col_w * len(header_cols) + 1)

    print("\n" + sep)
    print(f"PROJECTION TABLE  (raw % mean(std),  ×{N_seeds} random seeds)")
    print(sep)

    hdr = f"{'Projection Surface':<{struct_col}}"
    for col in header_cols:
        hdr += f"{col:>{col_w}}"
    print(hdr)
    print("─" * len(sep))

    for row_pos, (cname, orig_i) in enumerate(zip(ordered_names, reindex)):
        # ── individual cluster row ─────────────────────────────────────────
        row = f"{cname:<{struct_col}}"
        for pdb_name, metrics in pdb_targets.items():
            rf_m = metrics["raw_freq"]["mean"][orig_i]
            rf_s = metrics["raw_freq"]["std"][orig_i]
            cell = f"{rf_m*100:.1f}({rf_s*100:.1f})%"
            row += f"{cell:>{col_w}}"
        if null_rf_mean is not None:
            null_m    = null_rf_mean[orig_i]
            null_s    = null_rf_per_seed[:, orig_i].std()
            null_cell = f"{null_m*100:.1f}({null_s*100:.1f})%"
            row      += f"{null_cell:>{col_w}}"
        print(row)

        # ── HCR summary row after every 2 clusters, important go get std dev between folds ──────────
        if (row_pos + 1) % 2 == 0:
            hcr_num = (row_pos + 1) // 2
            row2    = f"{'  HCR-' + str(hcr_num):<{struct_col}}"
            grp     = [reindex[row_pos - 1], reindex[row_pos]]

            for pdb_name, metrics in pdb_targets.items():
                rf_per_seed  = metrics["raw_freq"]["per_seed"]
                hcr_per_seed = rf_per_seed[:, grp[0]] + rf_per_seed[:, grp[1]]
                hcr_m = hcr_per_seed.mean()
                hcr_s = hcr_per_seed.std()
                cell  = f"{hcr_m*100:.1f}({hcr_s*100:.1f})%"
                row2 += f"{cell:>{col_w}}"

            if null_rf_mean is not None:
                null_hcr_per_seed = null_rf_per_seed[:, grp[0]] + null_rf_per_seed[:, grp[1]]
                null_hcr_m        = null_hcr_per_seed.mean()
                null_hcr_s        = null_hcr_per_seed.std()
                null_cell         = f"{null_hcr_m*100:.1f}({null_hcr_s*100:.1f})%"
                row2             += f"{null_cell:>{col_w}}"

            print("─" * len(sep))
            print(row2)
            print("─" * len(sep))

    print(sep)

    # ── Stats table  ────────────────────────────────────────────
    print("\n" + sep)
    print("STATS TABLE")
    print(sep)

    stat_cols = ["Structure", "Closest match", "χ²", "P-value", "κ (closest)", "κ (HCR)"]
    sc        = [18, 28, 18, 18, 14, 10]
    print("".join(f"{c:<{w}}" for c, w in zip(stat_cols, sc)))
    print("─" * len(sep))

    for pdb_name, metrics in pdb_targets.items():
        rf_mean_arr = metrics["raw_freq"]["mean"]
        chi2_mean   = metrics["chi2"]["mean"]
        chi2_std    = metrics["chi2"]["std"]
        p_mean      = metrics["p_val"]["mean"]
        p_std       = metrics["p_val"]["std"]
        kappa_cm    = metrics.get("kappa_closest", {}).get("mean", float("nan"))
        kappa_hcr   = metrics.get("kappa_hcr",     {}).get("mean", float("nan"))

        best_row  = int(np.argmax([rf_mean_arr[i] for i in reindex]))
        best_name = ordered_names[best_row]
        hcr_tag   = f"(HCR-{best_row // 2 + 1})"

        vals = [
            name_map.get(pdb_name, pdb_name),
            f"{best_name} {hcr_tag}",
            f"{chi2_mean:.2f}({chi2_std:.2f})",
            f"{p_mean:.4f}({p_std:.4f})",
            f"{kappa_cm:.2f}" if not np.isnan(kappa_cm) else "—",
            f"{kappa_hcr:.2f}" if not np.isnan(kappa_hcr) else "—",
        ]
        print("".join(f"{v:<{w}}" for v, w in zip(vals, sc)))

    print(sep)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    #Make sure you are working in the Analysis directory
    
    N_seeds = 100
    SEEDS          = [(i+1) for i in range(N_seeds)]
    PCA_COMPONENTS = 90


    PROTEIN_PAIRS = [
        ("pickled_data/kaib_af.pkl", "pickled_data/kaib_og.pkl"),
        ("pickled_data/kaib_af.pkl", "pickled_data/kaib_extra_af.pkl"),
        ("pickled_data/mad2_af.pkl", "pickled_data/mad2_og.pkl"),
        ("pickled_data/mad2_af.pkl", "pickled_data/mad2_extra_af.pkl"),
        ("pickled_data/rfah_af.pkl", "pickled_data/rfah_og.pkl")
    ]

    for af_pkl, new_pkl in PROTEIN_PAIRS:
        print(f"\n{'#'*75}")
        print(f"# {af_pkl}  vs  {new_pkl}")
        print(f"{'#'*75}")

        seed_args = [(af_pkl, new_pkl, PCA_COMPONENTS, s) for s in SEEDS]

        all_results = []
        for args in seed_args:
            result = run_single_seed(args)
            seed = args[3]
            print(f"  seed {seed} done")
            all_results.append(result)

        

        _, target_names_original = all_results[0]
        aggregated = aggregate_results(all_results)

        # Pick cluster display order based on what protein runs, change names and make more elifs if you run other proteins
        if "kaib" in af_pkl.lower():
            cluster_order = order_kaib
        elif "rfah" in af_pkl.lower():
            cluster_order = order_rfah    
        else:
            cluster_order = order_mad2

        print_aggregated(aggregated, target_names_original, cluster_order, N_seeds)
        