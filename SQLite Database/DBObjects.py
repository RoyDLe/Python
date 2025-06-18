class Aircraft(): 
    def __init__(self, cargoId, flightId, cargoType, location):
        self.cargoId = cargoId
        self.flightId = flightId
        self.cargoType = cargoType
        self.location = location
    
    def get_fields():
        return "(cargoId INT UNIQUE PRIMARY KEY,\
            flightId INT,\
            cargoType TEXT,\
            location TEXT,\
            FOREIGN KEY(flightId) REFERENCES flights(flightId))"
    
    def get_insert_columns(self):
        return "(cargoId, flightId, cargoType, location) VALUES(?, ?, ?, ?)"
    
    def to_tuple(self):
        return (self.cargoId, self.flightId, self.cargoType, self.location)
    
class Order():
    def __init__(self, orderId, flightId, origin, destination, weight, volume):
        self.orderId = orderId 
        self.flightId = flightId
        self.origin = origin
        self.destination = destination
        self.weight = weight
        self.volume = volume
        
    def get_fields():
        return "(orderId INT UNIQUE PRIMARY KEY, \
            flightId INT, \
            origin TEXT, \
            destination TEXT, \
            weight REAL,\
            volume REAL,\
            FOREIGN KEY(flightId) REFERENCES flights(flightId))"
    
    def get_insert_columns(self):
        return "(orderId, flightId, origin, destination, weight, volume) VALUES(?, ?, ?, ?, ?, ?)"
    
    def to_tuple(self):
        return(self.orderId, self.flightId, self.origin, self.destination, self.weight, self.volume)
        

class Flight():
    def __init__(self, flightId, route, status):
        self.flightId = flightId
        self.route = route
        self.status = status 
        
    def get_fields():
        return "(flightId INT UNIQUE PRIMARY KEY, \
            route TEXT, \
            status TEXT)"
    
    def get_insert_columns(self):
        return "(flightId, route, status) VALUES(?, ?, ?)"
    
    def to_tuple(self):
        return(self.flightId, self.route, self.status)

        

        
    
            
        
        
        
        

