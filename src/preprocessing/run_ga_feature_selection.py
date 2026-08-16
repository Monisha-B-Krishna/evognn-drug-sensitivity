"""
run_ga_feature_selection.py (MULTI-OBJECTIVE: predictive performance + network connectivity)
Genetic Algorithm for gene feature selection. Fitness now combines two
objectives:
  1. Predictive performance: F1-macro of a logistic regression trained on
     the selected genes (as before)
  2. Network connectivity: fraction of selected genes that have at least
     one PPI edge to another selected gene

This directly addresses a discovered gap: pure predictive-performance
optimization produced a gene set with 73% fully isolated nodes in the PPI
graph (219/300 genes with zero connections) - a GNN would have almost no
graph structure to actually use. Adding connectivity as a co-objective
makes the GA search for genes that are BOTH predictive AND biologically
networked, which is the actual premise of the EvoGNN framework.

Input : data/processed/ga_sample.parquet
        data/processed/ppi_edges_filtered.csv
Output: data/processed/ga_selected_genes.csv
        data/processed/ga_convergence.csv
        results/checkpoints/ga_checkpoint_latest.pkl
"""
import os
import sys
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config_loader import load_config

GENE_BUDGET = 300
W_F1 = 0.7          # weight on predictive performance
W_CONNECTIVITY = 0.3  # weight on network connectivity


def main():
    cfg = load_config()
    ga_cfg = cfg["ga"]

    print("Loading GA sample...")
    df = pd.read_parquet("data/processed/ga_sample.parquet")

    non_gene_cols = {"SANGER_MODEL_ID", "CANCER_TYPE", "DRUG_ID", "DRUG_NAME", "LN_IC50", "label"}
    gene_cols = [c for c in df.columns if c not in non_gene_cols]
    n_genes = len(gene_cols)
    print(f"  Sample shape: {df.shape}, genes available: {n_genes}")
    print(f"  Gene budget per individual: {GENE_BUDGET}")
    print(f"  Fitness weights: F1={W_F1}, Connectivity={W_CONNECTIVITY}")

    X = df[gene_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=cfg["training"]["random_seed"]
    )
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # --- Precompute PPI adjacency, restricted to genes available for GA selection ---
    print("Building PPI adjacency lookup for connectivity scoring...")
    ppi_edges = pd.read_csv(cfg["paths"]["processed_ppi_edges_filtered"])
    gene_to_idx = {gene: i for i, gene in enumerate(gene_cols)}

    adjacency = {i: set() for i in range(n_genes)}
    edges_used = 0
    for g1, g2 in zip(ppi_edges["gene1"], ppi_edges["gene2"]):
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i1, i2 = gene_to_idx[g1], gene_to_idx[g2]
            adjacency[i1].add(i2)
            adjacency[i2].add(i1)
            edges_used += 1
    print(f"  Adjacency built from {edges_used} usable edges among {n_genes} candidate genes")

    def connectivity_score(selected_set):
        non_isolated = sum(1 for g in selected_set if adjacency[g] & selected_set)
        return non_isolated / len(selected_set)

    checkpoint_dir = "results/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "ga_checkpoint_latest.pkl")

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def init_individual():
        return creator.Individual(random.sample(range(n_genes), GENE_BUDGET))

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        idx = list(individual)
        X_train_sel = X_train[:, idx]
        X_val_sel = X_val[:, idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_val_scaled = scaler.transform(X_val_sel)

        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_val_scaled)
        f1 = f1_score(y_val, preds, average="macro")

        conn = connectivity_score(set(individual))

        combined = W_F1 * f1 + W_CONNECTIVITY * conn
        return (combined,)

    def cx_fixed_size(ind1, ind2):
        pool = list(set(ind1) | set(ind2))
        random.shuffle(pool)

        def build_child(preferred_pool):
            child = list(dict.fromkeys(preferred_pool))[:GENE_BUDGET]
            while len(child) < GENE_BUDGET:
                candidate = random.randrange(n_genes)
                if candidate not in child:
                    child.append(candidate)
            return child

        new1 = build_child(pool)
        new2 = build_child(pool[::-1])
        ind1[:] = new1
        ind2[:] = new2
        return ind1, ind2

    def mut_fixed_size(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                candidate = random.randrange(n_genes)
                attempts = 0
                while candidate in individual and attempts < 20:
                    candidate = random.randrange(n_genes)
                    attempts += 1
                individual[i] = candidate
        return (individual,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cx_fixed_size)
    toolbox.register("mutate", mut_fixed_size, indpb=ga_cfg["mutation_prob"])
    toolbox.register("select", tools.selTournament, tournsize=ga_cfg["tournament_size"])

    # Fresh run recommended — fitness function fundamentally changed, so a
    # resumed population's fitness values from the old objective wouldn't
    # be comparable. Uncomment below to allow resuming anyway if you prefer.
    # if os.path.exists(checkpoint_path):
    #     ...

    population = toolbox.population(n=ga_cfg["population_size"])
    start_gen = 0
    fitness_history = []
    print("Starting fresh GA run with connectivity-aware fitness")

    print("Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(start_gen, ga_cfg["generations"]):
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < ga_cfg["crossover_prob"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            if mutant.fitness.valid:
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        fits = [ind.fitness.values[0] for ind in population]
        best_ind_this_gen = population[fits.index(max(fits))]
        best_fitness = max(fits)
        avg_fitness = sum(fits) / len(fits)

        # Report the raw components too, not just the combined score
        best_conn = connectivity_score(set(best_ind_this_gen))

        fitness_history.append({
            "generation": gen, "best_combined": best_fitness,
            "avg_combined": avg_fitness, "best_connectivity": best_conn
        })
        print(f"Gen {gen}: best_combined={best_fitness:.4f}, avg_combined={avg_fitness:.4f}, "
              f"best_connectivity={best_conn:.3f}")

        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "population": population, "generation": gen,
                "fitness_history": fitness_history
            }, f)

    best_ind = tools.selBest(population, 1)[0]
    selected_genes = [gene_cols[i] for i in best_ind]

    # Recompute final F1 alone (not combined) for honest reporting
    idx = list(best_ind)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[:, idx])
    X_val_scaled = scaler.transform(X_val[:, idx])
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X_train_scaled, y_train)
    final_f1 = f1_score(y_val, clf.predict(X_val_scaled), average="macro")
    final_conn = connectivity_score(set(best_ind))

    pd.DataFrame({"gene": selected_genes}).to_csv("data/processed/ga_selected_genes.csv", index=False)
    pd.DataFrame(fitness_history).to_csv("data/processed/ga_convergence.csv", index=False)

    print(f"\n{'='*50}")
    print("GA feature selection complete (connectivity-aware)!")
    print(f"    Genes selected      : {len(selected_genes)} / {n_genes}")
    print(f"    Final F1-macro      : {final_f1:.4f}")
    print(f"    Final connectivity  : {final_conn:.1%} of genes non-isolated")
    print(f"    Combined fitness    : {best_ind.fitness.values[0]:.4f}")
    print(f"    Saved genes to      : data/processed/ga_selected_genes.csv")
    print(f"    Saved history to    : data/processed/ga_convergence.csv")
    print('='*50)


if __name__ == '__main__':
    main()
