import sqlite3 as db
from sqlite3 import Error

from exp_config import load_cfg
import numpy as np


class StoringExperience:
    """
    Class used to interface with the db and in which to store accuracy/loss values and tuning informations
    """
    def __init__(self):
        """
        initialise attributes in which to store into the db operations of creation and cleaning of tables
        """
        cfg = load_cfg()
        self.db_name = "{}/database/experience.db".format(cfg.name)
        self.destroy1 = """
            DROP TABLE IF EXISTS ranking
        """
        self.destroy2 = """
            DROP TABLE IF EXISTS experience
        """
        self.create1 = """
            CREATE TABLE IF NOT EXISTS ranking (
            id integer PRIMARY KEY,
            score real)
        """
        self.create2 = """
            CREATE TABLE IF NOT EXISTS experience (
            id integer PRIMARY KEY,
            evidence text)
        """

    def connection(self):
        """
        method used to interface with the database
        :return: connection with db
        """
        conn = None
        try:
            conn = db.connect(self.db_name)
            return conn
        except Error as e:
            print(e)
        return conn

    def create_db(self):
        """
        method used for creating the necessary tables in which to store the informations,
        executing the initially defined transactions
        """
        conn = self.connection()
        c = conn.cursor()
        try:
            c.execute(self.destroy1)
            c.execute(self.destroy2)
            c.execute(self.create1)
            c.execute(self.create2)
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def insert_ranking(self, score):
        """
        method used to insert accuracy and score values into the db
        """
        conn = self.connection()
        c = conn.cursor()
        print("[DEBUG] Inserting score into db: ", score)
        try:
            c.execute('INSERT INTO ranking (score) VALUES (' + str(score) + ')')
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def insert_evidence(self, evidence):
        """
        method used to insert evidences into the db.
        each evidence has anomaly of the network, a possible solution and a
        boolean that indicates if there was an improvement
        """
        conn = self.connection()
        c = conn.cursor()
        try:
            c.execute('INSERT INTO experience (evidence) VALUES ("' + str(evidence) + '")')
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def formatting(self, res):
        """
        method used to obtain score values into list
        :param res: list of lists, each of which will contain two values, one of acc and one of loss
        :return: list containing score values
        """
        scores = []
        for i in res:
            scores.append(i[1])
        return scores

    def get(self):
        """
        method used to obtain acc and loss values stored in the db
        :return: list of loss and acc values
        """
        conn = self.connection()
        c = conn.cursor()
        c.execute("SELECT * FROM ranking")
        res = self.formatting(c.fetchall())
        conn.close()
        print("[DEBUG] Getting score from db: ", res)
        return res


if __name__ == '__main__':
    se = StoringExperience()
    se.create_db()
    se.insert_ranking(0.7225, 0.7423)
    se.insert_ranking(0.7325, 0.7113)
    se.insert_ranking(0.7895, 0.4353)
    acc, loss = se.get()

    print(acc)
    print()
    print(loss)
