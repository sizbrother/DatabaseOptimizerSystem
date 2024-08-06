from sqlglot import exp, parse_one, condition
from sqlglot.expressions import Not, Or, And, Expression
import custom_ast
import queries
import transformations
import query_executor
import matplotlib.pyplot as plt
import re
import csv
import unittest
import pdb
import random
import json


class ASTNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children is not None else []
        self.visited = False

    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        if self.value != other.value or len(self.children) != len(other.children):
            return False
        return all(c1 == c2 for c1, c2 in zip(self.children, other.children))

    def __hash__(self):
        return hash((self.value, tuple(hash(child) for child in self.children)))
    


def plot_query_costs(query_plans):
    cost_pattern = r"cost=([\d.]+)\.\."
    costs = []

    for query in query_plans:
        if query is None:
            print("Warning: One of the queries is None.")
            costs.append(None)
            continue
        found_costs = re.findall(cost_pattern, query)
        if found_costs:
            print(float(found_costs[0]))
            costs.append(float(found_costs[0]))
        else:
            costs.append(None)

    valid_costs = [cost for cost in costs if cost is not None]
    queries = [f'Query {i+1}' for i in range(len(valid_costs))]

    plt.figure(figsize=(10, 5))
    plt.bar(queries, valid_costs, color='skyblue')
    plt.xlabel('Queries')
    plt.ylabel('Total Cost')
    plt.title('Total Costs of Each Query')
    if valid_costs:  
        plt.ylim(min(valid_costs) - 100, max(valid_costs) + 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()



def save_to_csv(sql_strings, query_plans, csv_file="query_plans.csv"):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SQL Query", "Query Plan"])  # Write header
        for sql_str, plan in zip(sql_strings, query_plans):
            writer.writerow([sql_str, plan])



def read_from_csv(csv_file):
    sql_strings = []
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            sql_strings.append(row[0]) 
    return sql_strings



def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


'''
def ast_generations(sql):

    expression_tree = parse_one(sql)

    where_ast = custom_ast.generate_ast(sql)
    #custom_ast.print_ast(where_ast)
    #print()

    
    orig_where = custom_ast.ast_to_query(where_ast)
    expr = transformations.transform_ast(where_ast)
    
    query_plans = set()
    sql_strings = set()

    #query_plans.append(query_executor.get_query_plan(sql))
    #print(query_executor.get_query_plan(sql))

    for tree in expr:
        #custom_ast.print_ast(tree)
        #print()
        where_clause = custom_ast.ast_to_query(tree)

        if where_clause != orig_where:
            transformed_tree = replace_where(expression_tree, where_clause)
            sql_string = transformed_tree.sql()
            #query_plans.add(query_executor.get_query_plan(sql_string))
            #print(query_executor.get_query_plan(sql_string))
            sql_strings.add(sql_string)
            #print(sql_string)

    sql_strings.add(sql)
 
    return list(sql_strings), list(query_plans)
'''

def replace_where(expression_tree, new_where_condition):
    
    expression_tree.find(exp.Where).replace(new_where_condition)
    return expression_tree



def apply_transformation_to_tree(node, target_node, transformation):
    """Recursively apply the transformation to the target node in the tree."""
    if node == target_node:
        return transformation
    
    if isinstance(node, Expression):
        new_args = {
            key: apply_transformation_to_tree(value, target_node, transformation)
            for key, value in node.args.items()
        }
        
        if type(node) == Not:
            return Not(**new_args)
        elif type(node) == Or:
            return Or(**new_args)
        elif type(node) == And:
            return And(**new_args)
        else:
            return type(node)(**new_args)
        
    return node



def ast_transform(sql):

    where_ast = custom_ast.generate_ast(sql)

    nodes_transformations = transformations.generate_transformations_list(where_ast)

    node, transformation = apply_random_transformation(nodes_transformations)

    if node and transformation:
        transformed_tree = apply_transformation_to_tree(where_ast, node, transformation)
        #print(f"Applied transformation: {transformation} to node: {node}")
        return transformed_tree, transformation
    else:
        print("No transformation applied.")
        return where_ast, None


"""
def apply_random_transformation(nodes_transformations):
    Pick a random node that has a transformation and apply it.
    transformable_nodes = [
        (node, transformations) for node, transformations in nodes_transformations
        if transformations['double_negatives'] or transformations['de_morgans'] or transformations['in_conversion'] or transformations['flip_comparison']
    ]
    if not transformable_nodes:
        #print("Nodes Trans: " + str(nodes_transformations))
        #print("No transformable nodes found.")
        return None, None
    
    node, transformations = random.choice(transformable_nodes)
    
    non_empty_transformations = {
        key: val for key, val in transformations.items() if val
    }
    
    chosen_type = random.choice(list(non_empty_transformations.keys()))
    
    chosen_transformation = random.choice(non_empty_transformations[chosen_type])

    return node, chosen_transformation
"""

def apply_random_transformation(nodes_transformations):
    """Pick a random type of transformation and apply it to a random node that supports it."""
    transformations_by_type = {
        'double_negatives': [],
        'de_morgans': [],
        'in_conversion': [],
        'flip_comparison': []
    }
    
    for node, transformations in nodes_transformations:
        for t_type, t_list in transformations.items():
            if t_list:
                transformations_by_type[t_type].append((node, t_list))
    
    non_empty_types = {t_type: t_list for t_type, t_list in transformations_by_type.items() if t_list}

    if not non_empty_types:
        return None, None

    chosen_type = random.choice(list(non_empty_types.keys()))
    print(chosen_type)
    
    transformable_nodes = non_empty_types[chosen_type]
    
    node, transformations_list = random.choice(transformable_nodes)
    chosen_transformation = random.choice(transformations_list)
    
    return node, chosen_transformation



def ast_walk(iterations):
    cost_pattern = r"cost=([\d.]+)\.\."
    query_trees = {}

    with open('query_plans.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Query Number', 'Iteration', 'Query', 'Plan', 'Cost', 'Transformation'])

        for query_number in range(1, 23):  # Iterate over queries q1 to q22
            print(query_number)

            if query_number == 15:
                continue

            file_path = f"queries/tpch-q{query_number}.sql"

            with open(file_path, "r") as query_file:
                sql = query_file.read()
            
            orig_sql = sql
            query_trees[query_number] = set()
            count = 0

            sql = orig_sql

            for i in range(iterations):


                try:
                    try:
                        expression_tree = parse_one(sql)
                    except:
                        print("RecursionError encountered. Skipping transformation.")
                        sql = orig_sql
                        continue

                    transform, transformation = ast_transform(sql)

                    if not transform:
                        continue

                    try: 
                        transformed_tree = replace_where(expression_tree, transform)
                    except RecursionError:
                        print("RecursionError encountered. Skipping transformation.")
                        sql = orig_sql
                        continue
                    except:
                        expression_tree = parse_one(orig_sql)
                        transform, transformation = ast_transform(orig_sql)
                        transformed_tree = replace_where(expression_tree, transform)

                    sql = transformed_tree.sql()
                    print(sql)

                    if sql in query_trees[query_number]:
                        continue
                    else:
                        query_trees[query_number].add(sql)

                    plan = query_executor.get_query_plan(sql)

                    try:
                        plan = query_executor.get_query_plan(sql)
                        if plan is None:
                            print("Plan is None. Skipping.")
                            sql = orig_sql
                            continue

                        found_costs = re.findall(cost_pattern, plan)
                        if not found_costs:
                            print("No costs found in the plan. Skipping.")
                            sql = orig_sql
                            continue

                        cost = float(found_costs[0])
                    except Exception as e:
                        print(f"Error encountered while getting query plan or parsing costs: {e}. Skipping.")
                        sql = orig_sql
                        continue

                    cost = float(found_costs[0])
                    count += 1

                    writer.writerow([query_number, count, sql, plan, cost, transformation])
                except:
                    sql = orig_sql
                    continue



config = read_config('config.json')
iterations = config.get('iterations', 10)

ast_walk(iterations)