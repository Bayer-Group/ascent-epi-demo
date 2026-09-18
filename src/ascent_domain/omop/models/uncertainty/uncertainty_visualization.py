import logging
import os
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def _plotting():
    """Import the plotting stack on first use.

    Both functions here are only reached from uncertainty_processing under
    ``if plot and out_folder``, which production never sets -- but the imports
    were at module scope, so every container start paid for matplotlib and
    seaborn and the Agg backend was configured process-wide as a side effect of
    an import.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")  # non-interactive; avoids tkinter threading errors
    import matplotlib.pyplot as plt
    import seaborn as sns

    return plt, sns


def plot_uncertainty_decomposition(uncertainty, interpretations, results, out_folder):
    """
    Generate and save visualization plots for uncertainty decomposition

    Args:
        uncertainty: Dictionary containing uncertainty metrics and matrices
        interpretations: List of interpretation texts
        results: List of result texts
        out_folder: Folder to save output files
    """
    plt, sns = _plotting()
    try:
        # Ensure output directory exists
        out_path = Path(out_folder)
        out_path.mkdir(parents=True, exist_ok=True)

        # Get matrices from uncertainty
        w_ii = uncertainty['matrices']['W_II']
        w_rr = uncertainty['matrices']['W_RR']
        w_ir = uncertainty['matrices']['W_IR']
        w_full = uncertainty['matrices']['W_full']

        # Get graphs from uncertainty
        g_ii = uncertainty['graphs']['G_II']
        g_rr = uncertainty['graphs']['G_RR']

        # 1. Plot interpretation similarity graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(g_ii, seed=42)  # For reproducible layout

        # Draw nodes
        nx.draw_networkx_nodes(g_ii, pos, node_size=700, node_color="lightblue")

        # Draw node labels
        labels = {i: f"I{i + 1}" for i in range(len(interpretations))}
        nx.draw_networkx_labels(g_ii, pos, labels=labels, font_size=10)

        # Draw edges with width proportional to weight (skip self-loops for clarity)
        edges = [(u, v) for u, v, d in g_ii.edges(data=True) if u != v]
        weights = [g_ii[u][v]["weight"] * 3 for u, v in edges]  # Scale for visibility

        nx.draw_networkx_edges(g_ii, pos, edgelist=edges, width=weights, alpha=0.7)

        # Add edge labels
        edge_labels = {(u, v): f"{g_ii[u][v]['weight']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(g_ii, pos, edge_labels=edge_labels, font_size=8)

        plt.title(f"Interpretation Similarity Graph (H_I: {uncertainty['entropies']['H_I']:.4f})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(Path(out_folder) / "interpretation_graph.png", dpi=300)
        plt.close()

        # 2. Plot result similarity graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(g_rr, seed=42)  # For reproducible layout

        # Draw nodes
        nx.draw_networkx_nodes(g_rr, pos, node_size=700, node_color="lightgreen")

        # Draw node labels
        labels = {i: f"R{i + 1}" for i in range(len(results))}
        nx.draw_networkx_labels(g_rr, pos, labels=labels, font_size=10)

        # Draw edges with width proportional to weight (skip self-loops for clarity)
        edges = [(u, v) for u, v, d in g_rr.edges(data=True) if u != v]
        weights = [g_rr[u][v]["weight"] * 3 for u, v in edges]  # Scale for visibility

        nx.draw_networkx_edges(g_rr, pos, edgelist=edges, width=weights, alpha=0.7)

        # Add edge labels
        edge_labels = {(u, v): f"{g_rr[u][v]['weight']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(g_rr, pos, edge_labels=edge_labels, font_size=8)

        plt.title(f"Result Similarity Graph (H_R: {uncertainty['entropies']['H_R']:.4f})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(Path(out_folder) / "result_graph.png", dpi=300)
        plt.close()

        # 3. Plot full system graph
        plt.figure(figsize=(12, 10))

        # Create a new graph for visualization that combines interpretations and results
        G_full = nx.Graph()

        # Add interpretation nodes
        for i in range(len(interpretations)):
            G_full.add_node(f"I{i}", type="interpretation")

        # Add result nodes
        for i in range(len(results)):
            G_full.add_node(f"R{i}", type="result")

        # Add edges between interpretations
        for i in range(len(interpretations)):
            for j in range(i + 1, len(interpretations)):
                if w_ii[i, j] > 0:
                    G_full.add_edge(f"I{i}", f"I{j}", weight=w_ii[i, j])

        # Add edges between results
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if w_rr[i, j] > 0:
                    G_full.add_edge(f"R{i}", f"R{j}", weight=w_rr[i, j])

        # Add edges between interpretations and results (mapping)
        for i in range(len(interpretations)):
            for j in range(len(results)):
                if w_ir[i, j] > 0:
                    G_full.add_edge(f"I{i}", f"R{j}", weight=w_ir[i, j], style="dashed")

        # Position nodes using spring layout
        pos = nx.spring_layout(G_full, seed=42)

        # Draw interpretation nodes
        interp_nodes = [n for n in G_full.nodes() if n.startswith("I")]
        nx.draw_networkx_nodes(G_full, pos, nodelist=interp_nodes, node_color="lightblue", node_size=700)

        # Draw result nodes
        result_nodes = [n for n in G_full.nodes() if n.startswith("R")]
        nx.draw_networkx_nodes(G_full, pos, nodelist=result_nodes, node_color="lightgreen", node_size=700)

        # Draw node labels
        nx.draw_networkx_labels(G_full, pos, font_size=10)

        # Draw similarity edges (between interpretations or between results)
        similarity_edges = [(u, v) for u, v in G_full.edges()
                            if (u.startswith("I") and v.startswith("I")) or
                            (u.startswith("R") and v.startswith("R"))]

        similarity_weights = [G_full[u][v]["weight"] * 3 for u, v in similarity_edges]
        nx.draw_networkx_edges(G_full, pos, edgelist=similarity_edges, width=similarity_weights, alpha=0.7)

        # Draw mapping edges (between interpretations and results)
        mapping_edges = [(u, v) for u, v in G_full.edges()
                         if (u.startswith("I") and v.startswith("R")) or
                         (u.startswith("R") and v.startswith("I"))]

        nx.draw_networkx_edges(G_full, pos, edgelist=mapping_edges,
                               width=2, alpha=1.0, edge_color="red", style="dashed")

        # Add edge labels for similarity edges
        similarity_edge_labels = {(u, v): f"{G_full[u][v]['weight']:.2f}" for u, v in similarity_edges}
        nx.draw_networkx_edge_labels(G_full, pos, edge_labels=similarity_edge_labels, font_size=8)

        plt.title(f"Full System Graph (H_QIR: {uncertainty['entropies']['H_QIR']:.4f})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(Path(out_folder) / "full_system_graph.png", dpi=300)
        plt.close()

        # 4. Plot uncertainty decomposition
        plt.figure(figsize=(10, 6))

        # Get all entropy values
        H_I = uncertainty['entropies']['H_I']
        H_R_given_I = uncertainty['entropies']['H(R|I)']
        H_R = uncertainty['entropies']['H_R']
        H_I_R = uncertainty['entropies']['H_QIR']  # H(I,R) = H_QIR

        # Create bar plot with all four entropy measures
        labels = ['H(I)\nInterpretation\nAmbiguity',
                  'H(R|I)\nModel\nInstability',
                  'H(R)\nResult\nEntropy',
                  'H(I,R)\nTotal\nUncertainty']
        values = [H_I, H_R_given_I, H_R, H_I_R]
        colors = ['#4A90E2', '#50C878', '#FFB347', '#E74C3C']  # Blue, Green, Orange, Red

        bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add horizontal line showing chain rule: H(I,R) = H(I) + H(R|I)
        chain_rule_sum = H_I + H_R_given_I
        plt.axhline(y=chain_rule_sum, color='purple', linestyle='--', linewidth=2,
                   label=f'H(I) + H(R|I) = {chain_rule_sum:.4f}')

        plt.ylabel("Entropy (bits)", fontsize=12, fontweight='bold')
        plt.title("Uncertainty Decomposition - Complete View", fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(Path(out_folder) / "uncertainty_decomposition.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Plot similarity matrices as heatmaps
        plot_similarity_heatmaps(uncertainty, interpretations, results, out_folder)

        logger.info(f"All plots saved to {out_folder}")

    except ImportError:
        logger.error("Required plotting libraries for graphs (networkx) not available")


def plot_similarity_heatmaps(uncertainty, interpretations, results, out_folder):
    """
    Generate and save heatmap visualizations for similarity matrices

    Args:
        uncertainty: Dictionary containing uncertainty metrics and matrices
        interpretations: List of interpretation texts
        results: List of result texts
        out_folder: Folder to save output files
    """
    plt, sns = _plotting()
    # Get matrices
    w_ii = uncertainty['matrices']['W_II']
    w_rr = uncertainty['matrices']['W_RR']
    w_full = uncertainty['matrices']['W_full']

    # W_II heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(w_ii, annot=True, cmap="YlGnBu", vmin=0, vmax=1,
                xticklabels=[f"I{i + 1}" for i in range(len(interpretations))],
                yticklabels=[f"I{i + 1}" for i in range(len(interpretations))])
    plt.title("Interpretation Similarity Matrix (W_II)")
    plt.savefig(Path(out_folder) / "w_ii_heatmap.png", dpi=300)
    plt.close()

    # W_RR heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(w_rr, annot=True, cmap="YlGnBu", vmin=0, vmax=1,
                xticklabels=[f"R{i + 1}" for i in range(len(results))],
                yticklabels=[f"R{i + 1}" for i in range(len(results))])
    plt.title("Result Similarity Matrix (W_RR)")
    plt.savefig(Path(out_folder) / "w_rr_heatmap.png", dpi=300)
    plt.close()

    # W_full heatmap
    plt.figure(figsize=(10, 8))
    n_interp = len(interpretations)
    n_results = len(results)

    # Create labels for the full matrix
    labels = [f"I{i + 1}" for i in range(n_interp)] + [f"R{i + 1}" for i in range(n_results)]

    sns.heatmap(w_full, annot=False, cmap="YlGnBu", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels)

    # Add lines to separate interpretations and results
    plt.axhline(y=n_interp, color='r', linestyle='-')
    plt.axvline(x=n_interp, color='r', linestyle='-')

    plt.title("Full System Similarity Matrix (W)")
    plt.savefig(Path(out_folder) / "w_full_heatmap.png", dpi=300)
    plt.close()