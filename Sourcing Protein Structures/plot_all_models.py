import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mad2_csv_file = "Mad2_AFcluster_analysis.csv"
mad2_rmsd1_name = "rmsd_1DUJ"
mad2_rmsd2_name = "rmsd_1S2H"
mad2_tm1_name = "tm_1DUJ"
mad2_tm2_name = "tm_1S2H"
mad2_lddt1_name = "lddt_1DUJ"
mad2_lddt2_name= "lddt_1S2H"
mad2_ref1_name=  "Mad2-open"
mad2_ref2_name=  "Mad2-closed"

mad2_df = pd.read_csv(mad2_csv_file)
# tm_4KSOchainA,tm_5N8YchainG,rmsd_4KSOchainA,rmsd_5N8YchainG,lddt_4KSOchainA,lddt_5N8YchainG,mean_plddt
kaib_csv_file = "KaiB_AFcluster_analysis.csv"
kaib_rmsd1_name = "rmsd_4KSOchainA"
kaib_rmsd2_name = "rmsd_5N8YchainG"
kaib_tm1_name = "tm_4KSOchainA"
kaib_tm2_name = "tm_5N8YchainG"
kaib_lddt1_name = "lddt_4KSOchainA"
kaib_lddt2_name= "lddt_5N8YchainG"
kaib_ref1_name=  "KaiB-ground"
kaib_ref2_name=  "KaiB-FS"

kaib_df = pd.read_csv(kaib_csv_file)

# id,idx,tm_2OUGchainA,tm_6C6SchainD,rmsd_2OUGchainA,rmsd_6C6SchainD,lddt_2OUGchainA,lddt_6C6SchainD,mean_plddt
rfah_csv_file = "RfaH_analysis.csv"
rfah_rmsd1_name = "rmsd_2OUGchainA"
rfah_rmsd2_name = "rmsd_6C6SchainD"
rfah_tm1_name = "tm_2OUGchainA"
rfah_tm2_name = "tm_6C6SchainD"
rfah_lddt1_name = "lddt_2OUGchainA"
rfah_lddt2_name= "lddt_6C6SchainD"
rfah_ref1_name=  "RfaH-AIn"
rfah_ref2_name=  "RfaH-active"

rfah_df = pd.read_csv(rfah_csv_file)


# Plot stuff
cmap = "viridis" 
alpha = 0.7  # Opaqueness of scatters
vmin = 50
vmax = 90
arrowsize = 0.5
shrinkA = 0
shrinkB = 5

def plot_report_plots():
    # Define 3x2 subplots 
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharex=False, sharey=False)
    ax1 = axes[0, 0]  
    ax2 = axes[0, 1]  
    ax3 = axes[1, 0]  
    ax4 = axes[1, 1]  
    ax5 = axes[2, 0]  
    ax6 = axes[2, 1]  

    # Make room for the colorbar on the right by adjusting the subplots to be a bit narrower
    fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.15)

    
    # ----- Plot tm1 vs tm2 -----
    mad2_config = {
        55: {"name": r"Mad2-$\alpha$1", "pos": (5, -5),  "arrow": False},
        81: {"name": r"Mad2-$\alpha$2", "pos": (5, 5),   "arrow": False},
        19: {"name": r"Mad2-$\alpha$3", "pos": (10, 25), "arrow": True},
        27: {"name": r"Mad2-$\alpha$4", "pos": (22, 5),  "arrow": True},
        87: {"name": r"Mad2-$\alpha$5", "pos": (20, -20), "arrow": True}
    }
    
    mad2_selection = list(mad2_config.keys())
    mad2_df_notselected = mad2_df[~mad2_df["idx"].isin(mad2_selection)]
    mad2_df_selected = mad2_df[mad2_df["idx"].isin(mad2_selection)]

    # ----- Plot tm1 vs tm2 -----
    # Plot all, but make the selection more visible with edgecolor and full opacity
    sc = ax1.scatter(mad2_df_notselected[mad2_tm1_name], mad2_df_notselected[mad2_tm2_name], c=mad2_df_notselected["mean_plddt"], cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
    sc2 = ax1.scatter(mad2_df_selected[mad2_tm1_name], mad2_df_selected[mad2_tm2_name], 
                      c=mad2_df_selected["mean_plddt"], cmap=cmap, alpha=1.0, 
                      edgecolor="black", linewidths=0.75, vmin=vmin, vmax=vmax, zorder=3)
    # Annotate the selection
    for _, row in mad2_df_selected.iterrows():
            c = mad2_config[row["idx"]]  
            
            ax1.annotate(
                c["name"], 
                xy=(row[mad2_tm1_name], row[mad2_tm2_name]), 
                xytext=c["pos"], 
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black", linewidth=arrowsize, shrinkA=shrinkA, shrinkB=shrinkB) if c["arrow"] else None,
                fontsize=9,
                va='center'
            )

    ax1.grid(True)
    ax1.axline((0, 0), slope=1, color="grey", linestyle=":", alpha=0.5) # x=y helper-line
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax1.set_xlabel("TM-score against " + mad2_ref1_name)
    ax1.set_ylabel("TM-score against " + mad2_ref2_name)
    ax1.set_title("TM-scores of AF models against Mad2 conformations")

    # ----- Plot lddt1 vs lddt2 -----
    mad2_config = {
        55: {"name": r"Mad2-$\alpha$1", "pos": (3, -5),  "arrow": False},
        81: {"name": r"Mad2-$\alpha$2", "pos": (3, 5),   "arrow": False},
        19: {"name": r"Mad2-$\alpha$3", "pos": (30, 10), "arrow": True},
        27: {"name": r"Mad2-$\alpha$4", "pos": (20, 0),  "arrow": True},
        87: {"name": r"Mad2-$\alpha$5", "pos": (-54, 15), "arrow": True}
    }
    
    mad2_selection = list(mad2_config.keys())
    mad2_df_notselected = mad2_df[~mad2_df["idx"].isin(mad2_selection)]
    mad2_df_selected = mad2_df[mad2_df["idx"].isin(mad2_selection)]

    # Plot all, but make the selection more visible with edgecolor and full opacity
    sc = ax2.scatter(mad2_df_notselected[mad2_lddt1_name], mad2_df_notselected[mad2_lddt2_name], c=mad2_df_notselected["mean_plddt"], cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
    sc2 = ax2.scatter(mad2_df_selected[mad2_lddt1_name], mad2_df_selected[mad2_lddt2_name], 
                      c=mad2_df_selected["mean_plddt"], cmap=cmap, alpha=1.0, 
                      edgecolor="black", linewidths=0.75, vmin=vmin, vmax=vmax, zorder=3)
    
    # Annotate the selection
    for _, row in mad2_df_selected.iterrows():
            c = mad2_config[row["idx"]]  
            
            ax2.annotate(
                c["name"], 
                xy=(row[mad2_lddt1_name], row[mad2_lddt2_name]), 
                xytext=c["pos"], 
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black", linewidth=arrowsize, shrinkA=shrinkA, shrinkB=shrinkB) if c["arrow"] else None,
                fontsize=9,
                va='center'
            )

    ax2.grid(True)
    ax2.axline((0, 0), slope=1, color="grey", linestyle=":", alpha=0.5) # x=y helper-line
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax2.set_xlabel("lDDT against " + mad2_ref1_name)
    ax2.set_ylabel("lDDT against " + mad2_ref2_name)
    ax2.set_title("lDDT of AF models against Mad2 conformations")

    # Same but for KaiB
        # ----- Plot tm1 vs tm2 -----
    kaib_config = {
        33: {"name": r"KaiB-$\alpha$1", "pos": (8, 15),  "arrow": True},
        102: {"name": r"KaiB-$\alpha$2", "pos": (2, -15),   "arrow": True},
        114: {"name": r"KaiB-$\alpha$3", "pos": (-45, -10), "arrow": True},
        48: {"name": r"KaiB-$\alpha$4", "pos": (-50, 20),  "arrow": True},
        72: {"name": r"KaiB-$\alpha$5", "pos": (15, 10), "arrow": True},
        94: {"name": r"KaiB-$\alpha$6", "pos": (15, 10), "arrow": True}
    }
    
    kaib_selection = list(kaib_config.keys())
    kaib_df_notselected = kaib_df[~kaib_df["idx"].isin(kaib_selection)]
    kaib_df_selected = kaib_df[kaib_df["idx"].isin(kaib_selection)]

    # Plot all, but make the selection more visible with edgecolor and full opacity
    sc = ax3.scatter(kaib_df_notselected[kaib_tm1_name], kaib_df_notselected[kaib_tm2_name], c=kaib_df_notselected["mean_plddt"], cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
    sc2 = ax3.scatter(kaib_df_selected[kaib_tm1_name], kaib_df_selected[kaib_tm2_name], 
                      c=kaib_df_selected["mean_plddt"], cmap=cmap, alpha=1.0, 
                      edgecolor="black", linewidths=0.75, vmin=vmin, vmax=vmax, zorder=3)
    
    # Annotate the selection
    for _, row in kaib_df_selected.iterrows():
            c = kaib_config[row["idx"]]  
            
            ax3.annotate(
                c["name"], 
                xy=(row[kaib_tm1_name], row[kaib_tm2_name]), 
                xytext=c["pos"], 
                textcoords="offset points",
                # Skapa arrowprops endast om "arrow" är True
                arrowprops=dict(arrowstyle="->", color="black", linewidth=arrowsize, shrinkA=shrinkA, shrinkB=shrinkB) if c["arrow"] else None,
                fontsize=9,
                va='center'
            )

    ax3.grid(True)
    ax3.axline((0, 0), slope=1, color="grey", linestyle=":", alpha=0.5) # x=y helper-line
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    ax3.set_xlabel("TM-score against " + kaib_ref1_name)
    ax3.set_ylabel("TM-score against " + kaib_ref2_name)
    ax3.set_title("TM-scores of AF models against KaiB conformations")

    # ----- Plot lddt1 vs lddt2 -----
    kaib_config = {
        33: {"name": r"KaiB-$\alpha$1", "pos": (10, 10),  "arrow": True},
        102: {"name": r"KaiB-$\alpha$2", "pos": (3, 3),   "arrow": False},
        114: {"name": r"KaiB-$\alpha$3", "pos": (3, 18), "arrow": True},
        48: {"name": r"KaiB-$\alpha$4", "pos": (-50, 15),  "arrow": True},
        72: {"name": r"KaiB-$\alpha$5", "pos": (15, 15), "arrow": True},
        94: {"name": r"KaiB-$\alpha$6", "pos": (-60, 3), "arrow": True}
    }
    
    kaib_selection = list(kaib_config.keys())
    kaib_df_notselected = kaib_df[~kaib_df["idx"].isin(kaib_selection)]
    kaib_df_selected = kaib_df[kaib_df["idx"].isin(kaib_selection)]

    # Plot all, but make the selection more visible with edgecolor and full opacity
    sc = ax4.scatter(kaib_df_notselected[kaib_lddt1_name], kaib_df_notselected[kaib_lddt2_name], c=kaib_df_notselected["mean_plddt"], cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
    sc2 = ax4.scatter(kaib_df_selected[kaib_lddt1_name], kaib_df_selected[kaib_lddt2_name], 
                      c=kaib_df_selected["mean_plddt"], cmap=cmap, alpha=1.0, 
                      edgecolor="black", linewidths=0.75, vmin=vmin, vmax=vmax, zorder=3)
    
    # Annotate the selection
    for _, row in kaib_df_selected.iterrows():
            c = kaib_config[row["idx"]]  
            
            ax4.annotate(
                c["name"], 
                xy=(row[kaib_lddt1_name], row[kaib_lddt2_name]), 
                xytext=c["pos"], 
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black", linewidth=arrowsize, shrinkA=shrinkA, shrinkB=shrinkB) if c["arrow"] else None,
                fontsize=9,
                va='center'
            )

    ax4.grid(True)
    ax4.axline((0, 0), slope=1, color="grey", linestyle=":", alpha=0.5) # x=y helper-line
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    ax4.set_xlabel("lDDT against " + kaib_ref1_name)
    ax4.set_ylabel("lDDT against " + kaib_ref2_name)
    ax4.set_title("lDDT of AF models against KaiB conformations")

    # Same but for RfaH
    # ----- Plot tm1 vs tm2 -----
    rfah_config = {
        1: {"name": r"RfaH-$\alpha$1", "pos": (5, 5),  "arrow": False},
        174: {"name": r"RfaH-$\alpha$2", "pos": (-50, 12),   "arrow": True},
        136: {"name": r"RfaH-$\alpha$3", "pos": (10, 20), "arrow": True},
        101: {"name": r"RfaH-$\alpha$4", "pos": (0, -30),  "arrow": True},
        102: {"name": r"RfaH-$\alpha$5", "pos": (20, -20), "arrow": True},
        103: {"name": r"RfaH-$\alpha$6", "pos": (20, 20), "arrow": True}
    }
    
    rfah_selection = list(rfah_config.keys())
    rfah_df_notselected = rfah_df[~rfah_df["idx"].isin(rfah_selection)]
    rfah_df_selected = rfah_df[rfah_df["idx"].isin(rfah_selection)]

    # ----- Plot tm1 vs tm2 -----
    # Plot all, but make the selection more visible with edgecolor and full opacity
    sc = ax5.scatter(rfah_df_notselected[rfah_tm1_name], rfah_df_notselected[rfah_tm2_name], c=rfah_df_notselected["mean_plddt"], cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
    sc2 = ax5.scatter(rfah_df_selected[rfah_tm1_name], rfah_df_selected[rfah_tm2_name], 
                      c=rfah_df_selected["mean_plddt"], cmap=cmap, alpha=1.0, 
                      edgecolor="black", linewidths=0.75, vmin=vmin, vmax=vmax, zorder=3)
    
    # Annotate the selection
    for _, row in rfah_df_selected.iterrows():
            c = rfah_config[row["idx"]]  
            
            ax5.annotate(
                c["name"], 
                xy=(row[rfah_tm1_name], row[rfah_tm2_name]), 
                xytext=c["pos"], 
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black", linewidth=arrowsize, shrinkA=shrinkA, shrinkB=shrinkB) if c["arrow"] else None,
                fontsize=9,
                va='center'
            )

    ax5.grid(True)
    ax5.axline((0, 0), slope=1, color="grey", linestyle=":", alpha=0.5) # x=y helper-line
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)

    ax5.set_xlabel("TM-score against " + rfah_ref1_name)
    ax5.set_ylabel("TM-score against " + rfah_ref2_name)
    ax5.set_title("TM-scores of AF models against RfaH conformations")

    # ----- Plot lddt1 vs lddt2 -----
    rfah_config = {
        1: {"name": r"RfaH-$\alpha$1", "pos": (5, 5),  "arrow": False},
        174: {"name": r"RfaH-$\alpha$2", "pos": (-50, 5),   "arrow": True},
        136: {"name": r"RfaH-$\alpha$3", "pos": (10, 10), "arrow": True},
        101: {"name": r"RfaH-$\alpha$4", "pos": (15, 5),  "arrow": True},
        102: {"name": r"RfaH-$\alpha$5", "pos": (20, -20), "arrow": True},
        103: {"name": r"RfaH-$\alpha$6", "pos": (-55, 10), "arrow": True}
    }
    
    rfah_selection = list(rfah_config.keys())
    rfah_df_notselected = rfah_df[~rfah_df["idx"].isin(rfah_selection)]
    rfah_df_selected = rfah_df[rfah_df["idx"].isin(rfah_selection)]

    # Plot all, but make the selection more visible with edgecolor and full opacity
    sc = ax6.scatter(rfah_df_notselected[rfah_lddt1_name], rfah_df_notselected[rfah_lddt2_name], c=rfah_df_notselected["mean_plddt"], cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
    sc2 = ax6.scatter(rfah_df_selected[rfah_lddt1_name], rfah_df_selected[rfah_lddt2_name], 
                      c=rfah_df_selected["mean_plddt"], cmap=cmap, alpha=1.0, 
                      edgecolor="black", linewidths=0.75, vmin=vmin, vmax=vmax, zorder=3)
    
    # Annotate the selection
    for _, row in rfah_df_selected.iterrows():
            c = rfah_config[row["idx"]]  
            ax6.annotate(
                c["name"], 
                xy=(row[rfah_lddt1_name], row[rfah_lddt2_name]), 
                xytext=c["pos"], 
                textcoords="offset points",
                # Skapa arrowprops endast om "arrow" är True
                arrowprops=dict(arrowstyle="->", color="black", linewidth=arrowsize, shrinkA=shrinkA, shrinkB=shrinkB) if c["arrow"] else None,
                fontsize=9,
                va='center'
            )

    ax6.grid(True)
    ax6.axline((0, 0), slope=1, color="grey", linestyle=":", alpha=0.5) # x=y helper-line
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_xlabel("lDDT against " + rfah_ref1_name)
    ax6.set_ylabel("lDDT against " + rfah_ref2_name)
    ax6.set_title("lDDT of AF models against RfaH conformations")
    # -----------------------------------------------------


    # ----- Common colorbar -----
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(sc2, cax=cbar_ax, label="Mean pLDDT")

    # ----- Top title -----
    fig.suptitle("Comparison of AF models against PDB structures for Mad2, KaiB and RfaH", fontsize=16, fontweight="bold", y=0.92)

    # ----- titles for each protein (row) -----
    fig.text(0.05, 0.857, "Mad2", fontsize=14, fontweight="bold", ha="left")
    fig.text(0.05, 0.58, "KaiB", fontsize=14, fontweight="bold", ha="left")
    fig.text(0.05, 0.305, "RfaH", fontsize=14, fontweight="bold", ha="left")


    # horizontal divider lines between rows
    height1 = 0.625
    height2 = 0.349

    divider_line = Line2D([0.05, 0.89], [height1, height1], color="black", linestyle="-", linewidth=1.0, alpha=0.9, transform=fig.transFigure)
    fig.add_artist(divider_line)

    divider_line = Line2D([0.05, 0.89], [height2, height2], color="black", linestyle="-", linewidth=1.0, alpha=0.9, transform=fig.transFigure)
    fig.add_artist(divider_line)

    fig.savefig("debugging.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_report_plots()