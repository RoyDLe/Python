import sqlite3

class DBManager():
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('PRAGMA foreign_keys = ON')

    def create_table(self, tableName, fields):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS " + tableName + fields)

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

    def drop_table(self, tableName):
        self.cursor.execute("DROP TABLE IF EXISTS " + tableName)
    
    def insert_data(self, tableName, columns, data):
        try:
            sql = "INSERT INTO " + tableName + columns
            self.cursor.execute(sql, data)
            self.conn.commit()
            
        except Exception as e:
            print("Something went wrong")
    
    def update_cell(self, tableName, columnName, value, condition):
        sql = f'UPDATE {tableName} SET {columnName} = {value} WHERE {condition}'
        self.cursor.execute(sql)
        self.conn.commit()

    def query_data(self, query):
        result = self.cursor.execute(query)
        self.conn.commit()

        return self.clean(result.fetchall())
    
    def clean(self, listOfTuples): 
        '''This function returns a list of values if only one column is referenced.
        For multiple columns a list of tuples is returned'''
        
        returnValue = []
        
        if len(listOfTuples) > 0 and len(listOfTuples[0]) == 1:
            for item in listOfTuples:
                returnValue.append(item[0])
            return returnValue
        
        else:
            return listOfTuples
                