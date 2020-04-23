#  Neo4j Python Driver connections 

from neo4j.v1 import GraphDatabase

class HPCJobDatabase(object):
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=1080)

    def close(self):
        self._driver.close()

    def query_small_set(self):
        with self._driver.session() as session:
            result = session.run("MATCH (n) OPTIONAL MATCH (n)-[r]->() RETURN n, r limit 10000000")
        return(result)

    