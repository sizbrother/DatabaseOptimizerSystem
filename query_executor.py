import psycopg2



def get_query_plan(query):

    db_params = {
    'database': 'postgres',
    'user': 'postgres',
    'password': 'student',
    'host': 'localhost',
    'port': '5432'
    }

    query = query + ";"

    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute("SET search_path TO tpch1g;")
    try:
        cur.execute("EXPLAIN" + query)
        plan = cur.fetchall()
        return '\n'.join(str(line[0]) for line in plan)
    except Exception as e:
        print("An error occurred:", e)
        return None
    finally:
        cur.close()
        conn.close()