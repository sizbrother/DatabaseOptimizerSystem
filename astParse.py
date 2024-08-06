import copy
import itertools
import sqlparse
import psycopg2
import re
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np



class ASTNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children is not None else []


'''
def tokenize(where_clause):

    tokens = []
    current_token = []
    operators = set(['AND', 'OR', 'NOT', '=', '<', '>', '>=', '<=', '(', ')', '!='])

    i = 0
    while i < len(where_clause):

        char = where_clause[i]

        if char.isalnum() or char == "'" or char == "_":
            current_token.append(char)
        # If we've reached a space then we will append
        # the token and need to append it to the list
        elif char.isspace():
            if current_token:
                token = ''.join(current_token)
                if token.upper() in operators:
                    tokens.append(token.upper())
                else:
                    tokens.append(token)
                current_token = [] # Reset current tokens
        else:
            if current_token:
                token = ''.join(current_token)
                if token.upper() in operators:
                    tokens.append(token.upper())
                else:
                    tokens.append(token)
                current_token = [] # Reset current tokens

            if (char in ['<', '>', '!', '='] and i + 1 < len(where_clause) and where_clause[i + 1] == '='):
                tokens.append(char + '=')
                i += 1
            else:
                tokens.append(char)
        i += 1

    if current_token:
        token = ''.join(current_token)
        if token.upper() in operators:
            tokens.append(token.upper())
        else:
            tokens.append(token)

    return tokens
'''
def tokenize(where_clause):
    tokens = []
    parsed = sqlparse.parse(where_clause)[0]
    
    # We will keep track of the last token to decide whether to merge with a period
    last_token = ""

    for token in parsed.flatten():
        # Remove whitespace
        if token.is_whitespace:
            continue
        
        if token.value == '.':
            # If the current token is a period, append it to the last token and prepare to merge the next token
            if tokens:
                last_token = tokens.pop() + '.'
        elif token.ttype in sqlparse.tokens.Punctuation and token.value != '.':
            # Skip other punctuation like commas
            continue
        else:
            # Normal token handling
            current_value = str(token)
            if last_token:  # There's a pending token to merge (from a period)
                current_value = last_token + current_value
                last_token = ""  # Reset the merge buffer

            if token.ttype in sqlparse.tokens.Whitespace:
                continue
            elif token.normalized in {'AND', 'OR', 'NOT', '=', '<', '>', '>=', '<=', '(', ')', '!='}:
                tokens.append(token.normalized)
            else:
                tokens.append(current_value)

    return tokens


def parse_tokens(tokens):
    operators = {'AND', 'OR', 'NOT', '=', '<', '>', '<=', '>=', '!=', 'IN'}
    precedence = {'OR': 1, 'AND': 2, 'NOT': 3, '=': 4, '<': 4, '>': 4, '<=': 4, '>=': 4, '!=': 4, 'IN': 4}
    stack = []
    output = []

    for token in tokens:
        if token in operators:
            while stack and stack[-1] != '(' and precedence[stack[-1]] >= precedence[token]:
                output.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove the '(' from the stack
        else:
            output.append(token)  # Append non-operator tokens (literals) to output

    while stack:
        output.append(stack.pop())

    # Convert output to AST
    ast_stack = []
    for item in output:
        if item in operators:
            if item == 'IN':
                values = []
                while ast_stack and ast_stack[-1] not in operators and ast_stack[-1] != '(':
                    values.append(ast_stack.pop())
                node = ASTNode('IN', children=values[::-1])  # Reverse the list for correct order
                ast_stack.append(node)
            else:
                right = ast_stack.pop()
                left = ast_stack.pop()
                node = ASTNode(item, children=[left, right])
                ast_stack.append(node)
        else:
            ast_stack.append(ASTNode(item))

    return ast_stack[0] if ast_stack else None


def print_ast(node, depth=0):
    if node:
        print('  ' * depth + node.value)
        for child in node.children:
            print_ast(child, depth+1)

def generate_ast(sql_query):
    where_clause = sql_query.lower().split("where")[1].strip()
    tokens = tokenize(where_clause)
    #print(tokens)
    root = parse_tokens(tokens)
    return root




def generate_equivalent_trees(node, depth=0, max_depth=3):
    """
    Generate semantically equivalent trees with a depth limit to prevent infinite recursion.
    """
    # Yield the node itself if it's a base case or leaf
    if depth >= max_depth or not node.children:
        yield node
        return

    # Introduce double negation at the current node
    double_negated = ASTNode('NOT', [ASTNode('NOT', [node])])
    yield double_negated

    # Recursively handle children if the node is a logical operator
    if node.value in ['AND', 'OR']:
        # Generate trees from children first
        for i, child in enumerate(node.children):
            for modified_child in generate_equivalent_trees(child, depth + 1, max_depth):
                new_children = node.children[:]
                new_children[i] = modified_child
                yield ASTNode(node.value, new_children)

        # Apply De Morgan's and recursively handle
        if node.value == 'AND':
            de_morgan_or = ASTNode('OR', [
                ASTNode('NOT', [node.children[0]]),
                ASTNode('NOT', [node.children[1]])
            ])
            for tree in generate_equivalent_trees(de_morgan_or, depth + 1, max_depth):
                yield tree
        elif node.value == 'OR':
            de_morgan_and = ASTNode('AND', [
                ASTNode('NOT', [node.children[0]]),
                ASTNode('NOT', [node.children[1]])
            ])
            for tree in generate_equivalent_trees(de_morgan_and, depth + 1, max_depth):
                yield tree

    # Handle NOT nodes specifically to prevent direct infinite recursion
    elif node.value == 'NOT':
        # Process the child with increased depth
        for tree in generate_equivalent_trees(node.children[0], depth + 1, max_depth):
            yield ASTNode('NOT', [tree])



def commutative_transformation(node):
    transformed_trees = []
    if node.value in ['AND', 'OR']:
        for permuted_children in itertools.permutations(node.children):
            if permuted_children != tuple(node.children):  # Exclude the original order
                transformed_trees.append(ASTNode(node.value, list(permuted_children)))
    return transformed_trees


def associative_transformation(node):
    transformed_trees = []

    if node.value not in ['AND', 'OR'] or len(node.children) <= 1:
        return transformed_trees

    from itertools import combinations

    # Generate all combinations of non-empty distinct partitions of the children
    for i in range(1, len(node.children)):
        for left_nodes in combinations(node.children, i):
            right_nodes = [x for x in node.children if x not in left_nodes]
            if left_nodes and right_nodes:
                new_left_node = ASTNode(node.value, list(left_nodes))
                new_right_node = ASTNode(node.value, list(right_nodes))
                new_tree = ASTNode(node.value, [new_left_node, new_right_node])

    return transformed_trees


'''
def distributive_transformation(node):
    transformed_trees = []
    if node.value == 'AND':
        or_children = [child for child in node.children if child.value == 'OR']
        non_or_children = [child for child in node.children if child.value != 'OR']
        
        for or_child in or_children:
            for or_grandchild in or_child.children:
                # Create a new AND node for each OR grandchild combined with the non-OR children
                new_and_children = [copy.deepcopy(or_grandchild)] + copy.deepcopy(non_or_children)
                new_and_node = ASTNode('AND', new_and_children)
                
                # Each of these new AND nodes will be an OR's child in the transformed tree
                transformed_trees.append(ASTNode('OR', [new_and_node]))
    return transformed_trees
'''



def transform_ast(node, transformations):
    """ Apply transformations to each node in the AST recursively. """
    if node is None:
        return []

    transformed_trees = [node]  # Start with the initial node.

    # Apply transformations at the current node
    for transform in transformations:
        transformed_nodes = transform(node)
        if not isinstance(transformed_nodes, list):
            transformed_nodes = [transformed_nodes]
        transformed_trees.extend(transformed_nodes)

    # Recursively apply transformations to each child
    for i, child in enumerate(node.children):
        child_transforms = transform_ast(child, transformations)
        for trans in child_transforms:
            new_node = copy.deepcopy(node)
            new_node.children[i] = trans
            transformed_trees.append(new_node)

    return transformed_trees



def ast_to_query(node):
    if not node:
        return ''

    if not node.children:
        return str(node.value)

    # Debug output to trace the content of node
    #print(f"Processing node with value: {node.value} and children count: {len(node.children)}")

    try:
        if node.value in {'AND', 'OR'}:
            if len(node.children) < 2:
                raise ValueError(f"Expected 2 children for '{node.value}' operator, got {len(node.children)}")
            operator = node.value
            left = ast_to_query(node.children[0])
            right = ast_to_query(node.children[1])
            return f'({left} {operator} {right})'

        if node.value == 'IN':
            if len(node.children) < 2:
                raise ValueError(f"Expected at least 2 children for 'IN' operator, got {len(node.children)}")
            column = ast_to_query(node.children[0])
            values = ', '.join(ast_to_query(child) for child in node.children[1:])
            return f'{column} IN ({values})'

        if node.value == 'BETWEEN':
            if len(node.children) < 3:
                raise ValueError(f"Expected 3 children for 'BETWEEN', got {len(node.children)}")
            column = ast_to_query(node.children[0])
            low = ast_to_query(node.children[1])
            high = ast_to_query(node.children[2])
            return f'{column} BETWEEN {low} AND {high}'

        if node.value == 'NOT':
            if len(node.children) < 1:
                raise ValueError(f"Expected 1 child for 'NOT' operator, got {len(node.children)}")
            condition = ast_to_query(node.children[0])
            return f'NOT ({condition})'

        # Handle other comparison operators
        if len(node.children) < 2:
            raise ValueError(f"Expected 2 children for comparison operator '{node.value}', got {len(node.children)}")
        column = ast_to_query(node.children[0])
        operator = node.value
        value = ast_to_query(node.children[1])
        return f'{column} {operator} {value}'
    except IndexError as e:
        #print(f"IndexError accessing children of node with value: '{node.value}'")
        raise e
    except Exception as ex:
        #print(f"Error processing node with value: '{node.value}'")
        raise ex




query_assoc = "SELECT * FROM Employees WHERE (Age > 30 AND Salary > 50000) AND (YearsAtCompany > 5 AND Department = 'Engineering')"
query_test = "EXPLAIN SELECT * FROM tpch1g.orders, tpch1g.customer WHERE (orders.o_totalprice > 50000 AND orders.o_orderkey > 1335420009) AND (customer.c_acctbal > 10000 AND customer.c_mktsegment = 'AUTOMOBILE')"
query = """
from
	lineitem,
	part
where
	(
		p_partkey = l_partkey
		and p_brand = ':1'
		and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
		and l_quantity >= :4 and l_quantity <= :4 + 10
		and p_size between 1 and 5
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)
	or
	(
		p_partkey = l_partkey
		and p_brand = ':2'
		and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
		and l_quantity >= :5 and l_quantity <= :5 + 10
		and p_size between 1 and 10
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)
	or
	(
		p_partkey = l_partkey
		and p_brand = ':3'
		and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
		and l_quantity >= :6 and l_quantity <= :6 + 10
		and p_size between 1 and 15
		and l_shipmode in ('AIR', 'AIR REG')
		and l_shipinstruct = 'DELIVER IN PERSON'
	)
"""

original_ast = generate_ast(query_test)
#print()
#print_ast(original_ast)


transformations_one = [
    commutative_transformation
    #associative_transformation
]

transformations_two = [
    associative_transformation
]

transformed_comm_ast = transform_ast(original_ast, transformations_one)
transformed_assoc_ast = transform_ast(original_ast, transformations_two)
transformed_demorgan = generate_equivalent_trees(original_ast)


for ast in transformed_assoc_ast:
    print("----")
    print(ast_to_query(ast))
    #print_ast(ast)

'''
for equivalent_tree in generate_equivalent_trees(original_ast):
    print_ast(equivalent_tree)
'''



def get_query_plan(db_params, transformation):


    explain_query = "EXPLAIN SELECT * FROM tpch1g.orders, tpch1g.customer WHERE " + transformation + ";"
    print(explain_query)

    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    
    try:
        cur.execute(explain_query)
        plan = cur.fetchall()
        return '\n'.join(line[0] for line in plan)
    except Exception as e:
        print("An error occurred:", e)
        return None
    finally:
        cur.close()
        conn.close()

def normalize_cost(explain_output):
    # Pattern to find the cost value which is usually in the format "cost=123.456.."
    cost_pattern = r'cost=(\d+\.\d+)'
    
    # Using search to find the first occurrence of the cost pattern
    match = re.search(cost_pattern, explain_output)
    
    if match:
        # If a match is found, return just the numeric cost value
        return match.group(1)  # This returns the first capture group from the regex
    else:
        # If no match is found, you might want to handle this situation, e.g., by returning None or an error message
        return "No cost found"


db_params = {
    'database': 'postgres',
    'user': 'postgres',
    'password': 'student',
    'host': 'localhost',
    'port': '5432'
}

def process_transformations_and_costs(transformations, db_params):
    transformed_costs = []
    for transformation in transformations:
        query = ast_to_query(transformation)
        execution_plan = get_query_plan(db_params, query)
        print(execution_plan)
        if execution_plan is None:
            #print("Skipping normalization due to failed execution plan.")
            transformed_costs.append("Error: Execution plan failed")
            continue
        normalized_plan = normalize_cost(execution_plan)
        transformed_costs.append(normalized_plan)
    return transformed_costs


x = process_transformations_and_costs(transformed_assoc_ast, db_params)
y = process_transformations_and_costs(transformed_comm_ast, db_params)
z = process_transformations_and_costs(transformed_demorgan, db_params)


transformation_types = ['Associative', 'Commutative', 'De Morgan']


x_count = Counter(x)
y_count = Counter(y)
z_count = Counter(z)

# Combine all unique costs and their counts for the graph
unique_costs = sorted(set(x + y + z), key=lambda x: float(x))  # All unique cost values sorted
indices = np.arange(len(unique_costs))  # the x locations for the groups

x_freq = [x_count[cost] for cost in unique_costs]
y_freq = [y_count[cost] for cost in unique_costs]
z_freq = [z_count[cost] for cost in unique_costs]

width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(indices - width, x_freq, width, label='Associative')
rects2 = ax.bar(indices, y_freq, width, label='Commutative')
rects3 = ax.bar(indices + width, z_freq, width, label='De Morgan')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Cost')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Cost by Transformation Type')
ax.set_xticks(indices)
ax.set_xticklabels(unique_costs)
ax.legend()

# Function to attach a text label above each bar, displaying its height
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()