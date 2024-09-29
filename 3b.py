# Experimnet 2 A(2)
#  program to randomly generate k-SAT problems

import random

def generate_k_sat(k, m, n):
    clauses = []
    variables = list(range(1, n+1))
    6
    for _ in range(m):
    clause = set()
    while len(clause) < k:
    variable = random.randint(1, n)
    negated = random.choice([True, False])
    if negated:
    variable = -variable
    clause.add(variable)
    clauses.append(clause)
    return clauses
    def print_k_sat(clauses):
    for i, clause in enumerate(clauses):
    clause_str = " v ".join(map(str, clause))
    print("(", clause_str, ")", end="")
    if i < len(clauses) - 1:
    print(" ∧ ", end="")
    print("\n")
    if __name__ == "__main__":
    k = int(input("Enter k (length of each clause): "))
    m = int(input("Enter m (number of clauses): "))
    n = int(input("Enter n (number of variables): "))
    7
    k_sat_problem = generate_k_sat(k, m, n)
    print("\nGenerated k-SAT problem:")
    print_k_sat(k_sat_problem)