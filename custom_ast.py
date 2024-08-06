from sqlglot import exp, parse_one, condition
from sqlglot.expressions import Expression
import copy
import sqlparse
import re
# This program using 3 steps:
# 1. Tokenize the SQL query, isolating the contents of the WHERE clause into operators and literals
# 2. Convert list of tokens into an AST 
# 3. Print the AST Tree


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
    


def tokenize(where_clause):
    tokens = []
    parsed = sqlparse.parse(where_clause)[0]
    last_token = ""

    for token in parsed.flatten():
        if token.is_whitespace:
            continue
        
        if token.value == '.':
            #Current token is period, append to last token
            if tokens:
                last_token = tokens.pop() + '.'
        elif token.ttype in sqlparse.tokens.Punctuation and token.value != '.' and token.value != '(' and token.value != ')':
            # Skip other punctuation like commas
            continue
        else:
            # Normal token handling
            current_value = str(token)
            if last_token:  # There's a pending token to merge (from a period)
                current_value = last_token + current_value
                last_token = ""

            if token.ttype in sqlparse.tokens.Whitespace:
                continue
            elif token.normalized in {'AND', 'OR', 'NOT', '=', '<', '>', '>=', '<=', '(', ')', '!=', '<>'}:
                tokens.append(token.normalized)
            else:
                tokens.append(current_value)

    return tokens


def parse_tokens(tokens):
    operators = {'AND', 'OR', 'NOT', '=', '<', '>', '<=', '>=', '!=', '<>'}
    special_operators = {'IN', 'BETWEEN', 'EXISTS', 'SUBSTRING'}
    precedence = {'OR': 1, 'AND': 2, 'NOT': 3, '=': 4, '<': 4, '>': 4, '<=': 4, '>=': 4, '!=': 4, '<>': 4, 'IN': 4, 'BETWEEN':4, 'EXISTS': 4, 'SUBSTRING': 4}
    stack = []
    output = []

    i = 0
    #print(tokens)
    while i < len(tokens):
        token = tokens[i]
        token_upper = token.upper()
        #print(token_upper)

        if token_upper in operators:
            while stack and stack[-1] != '(' and precedence.get(stack[-1], 0) >= precedence[token_upper]:
                output.append(stack.pop())
            stack.append(token_upper)
        elif token_upper in special_operators:
            stack.append(token_upper)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                #print(stack[-1])
                #print(token)
                #print(i)
                output.append(stack.pop())
            if stack and stack[-1] in special_operators:
                output.append(stack.pop())
            if not stack:
                break
            stack.pop()
        elif token_upper == 'SELECT':
            if stack[-1] == '(':
                stack.pop()
            subquery_tokens = []
            nested_level = 1
            i += 1
            while i < len(tokens) and nested_level > 0:
                if tokens[i] == '(':
                    nested_level += 1
                elif tokens[i] == ')':
                    nested_level -= 1
                subquery_tokens.append(tokens[i])
                i += 1
            subquery_tokens.pop()
            subquery_str = ' '.join(subquery_tokens)
            subquery_str = "select " + subquery_str
            print(subquery_str)
            parsed_ast = parse_one(subquery_str)
            stack.append(parsed_ast)
            i -= 1
        else:
            #print(token)
            output.append(token)

        i += 1

    while stack:
        output.append(stack.pop())

    # Convert output to AST
    ast_stack = []
    #print(tokens)
    print("Output: " + str(output))
    print()
    print()
    i = 0
    while i < len(output):
        item = output[i]
        print(item)
        if isinstance(item, Expression):
            ast_stack.append(item)
        elif item.upper() in operators or item.upper() in special_operators:
            if item.upper() == 'IN':
                values = []
                #print("Output: " + str(output))
                i += 1
                print("test " + output[i])
                while i < len(output) and output[i][0] == "'" and output[i][-1] == "'":
                    values.append(output[i])
                    i += 1
                print("Values: " + str(values))
                children = [ASTNode(value) for value in values]
                left = ast_stack.pop()
                children.insert(0, left)
                node = ASTNode('IN', children=children)
                ast_stack.append(node)
                continue
            elif item.upper() == 'BETWEEN':
                lower = ASTNode(output[i+1])  # lower limit
                if output[i+3] != "AND":
                    upper = ASTNode(output[i+3])  # upper limit
                else:
                    upper = ASTNode(output[i+2])
                left = ast_stack.pop()
                node = ASTNode('BETWEEN', children=[left, lower, upper])
                ast_stack.append(node)
                i += 4  # Skip the 'AND' and upper limit
                continue
            elif item.upper() == 'SUBSTRING':
                print("TEEE")
            elif item.upper() == 'NOT':
                if ast_stack:
                    operand = ast_stack.pop()
                    node = ASTNode('NOT', children=[operand])
                    ast_stack.append(node)
            elif item.upper() == 'EXISTS':
                subquery = ast_stack.pop()
                node = ASTNode('EXISTS', children=[subquery])
                ast_stack.append(node)
            else:
                if len(ast_stack) >= 2:
                    right = ast_stack.pop()
                    left = ast_stack.pop()
                    node = ASTNode(item, children=[left, right])
                    ast_stack.append(node)
                else:
                    print("Error: Stack doesn't have enough operands for the operator " + item)
        else:
            if i + 2 < len(output) and output[i+1] in {'+', '-', '*', '/'}:
                    left = ASTNode(item)
                    operator = output[i+1]
                    right = ASTNode(output[i+2])
                    node = ASTNode(operator, children=[left, right])
                    ast_stack.append(node)
                    i += 2
            else:
                ast_stack.append(ASTNode(item))
        i += 1

    return ast_stack[0] if ast_stack else None


def print_ast(node, depth=0):
    if hasattr(node, 'value'):
        print("  " * depth + str(node.value))
    else:
        print("  " * depth + str(node))
    
    if hasattr(node, 'children'):
        for child in node.children:
            print_ast(child, depth + 1)
    elif isinstance(node, list):
        for child in node:
            print_ast(child, depth + 1)



def isolate_where_clause(sql_query):
    where_pattern = re.compile(r"\bwhere\b", re.IGNORECASE)
    next_clause_pattern = re.compile(r"\b(group\s+by|order\s+by|having|limit)\b", re.IGNORECASE)

    where_match = where_pattern.search(sql_query)
    if not where_match:
        return None

    where_pos = where_match.end()
    
    next_clause_match = next_clause_pattern.search(sql_query, where_pos)
    if next_clause_match:
        next_clause_pos = next_clause_match.start()
    else:
        next_clause_pos = len(sql_query) 

    where_clause = sql_query[where_pos:next_clause_pos].strip()



    return where_clause


def consolidate_tokens(tokens):
    combined_tokens = []
    i = 0
    operators = {'AND', 'OR', 'IN'}
    
    while i < len(tokens):
        token_upper = tokens[i].upper()
        
        if token_upper == 'SELECT':
            while combined_tokens:
                if combined_tokens[-1].upper() not in operators:
                    combined_tokens.pop()
                elif combined_tokens[-1].upper() in operators:
                    combined_tokens.pop()
                    return combined_tokens
        else:
            combined_tokens.append(tokens[i])
        i += 1
    
    return combined_tokens



def find_positions_in_query(original_query, consolidated_tokens):
    original_query_clean = re.sub(r"\s+", "", original_query)
    consolidated_query_clean = ''.join(consolidated_tokens)

    start_pos = original_query_clean.find(consolidated_query_clean)

    if start_pos == -1:
        return None
    
    end_pos = start_pos + len(consolidated_query_clean)

    return start_pos, end_pos



def generate_ast(sql_query):

    #where_clause = isolate_where_clause(sql_query)
    #tokens = tokenize(where_clause)
    #root = parse_tokens(tokens)

    parsed = parse_one(sql_query)

    #print_ast(root)

    for expression in parsed.find_all(exp.Where):
        #print(expression)
        return expression
    
    return None




def ast_to_query(node):
    if not node:
        return ''
    

    if not hasattr(node, 'children'):
        return str(node.value)
    
    if hasattr(node, "expressions"):
        return(str(node))

    if not node.children:
        return str(node.value)

    try:
        if node.value in {'AND', 'OR'}:
            operator = node.value
            left = ast_to_query(node.children[0])
            right = ast_to_query(node.children[1])
            return f'({left} {operator} {right})'

        elif node.value == 'IN':
            column = ast_to_query(node.children[0])
            values = ', '.join(ast_to_query(child) for child in node.children[1:])
            return f"{column} IN ({values})"

        elif node.value == 'BETWEEN':
            column = ast_to_query(node.children[0])
            low = ast_to_query(node.children[1])
            high = ast_to_query(node.children[2])
            return f"{column} BETWEEN {low} AND {high}"
        
        elif node.value == 'EXISTS':
            subquery = node.children[0]
            return f"EXISTS ({subquery})"
        
        elif node.value == 'NOT':
            condition = ast_to_query(node.children[0])
            return f"NOT ({condition})"
        
        else:  # This handles '=', '!=', '<', '>', '<=', '>=' operators
            column = ast_to_query(node.children[0])
            value = ast_to_query(node.children[1])
            return f"{column} {node.value} {value}"
    except IndexError as e:
        raise IndexError(f"IndexError accessing children of node with value: '{node.value}': {e}")
    except Exception as ex:
        raise Exception(f"Error processing node with value: '{node.value}': {ex}")