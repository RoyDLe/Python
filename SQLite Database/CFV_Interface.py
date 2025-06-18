from DBManager import DBManager
from DBObjects import Flight

class CFV_Interface():
    def __init__(self, dbName):
        self.dbManager = DBManager(dbName)

    def search_manifest(self, flightnumber):
        return self.dbManager.query_data(f'SELECT orderId FROM orders WHERE flightId = {flightnumber}')

    def search_flight_for_route(self, departure, destination):
        return self.dbManager.query_data(f'SELECT flightId FROM flights WHERE route = "{departure}-{destination}"')
    
    def search_unassigned_orders(self): # Orders are unassigned if their flightId is NULL
        return self.dbManager.query_data('SELECT orderId FROM orders WHERE flightId IS NULL') 

    def search_available_planes_for_airport(self, airport_code):
        return self.dbManager.query_data(f'SELECT cargoId FROM cargo WHERE location = "{airport_code}" AND flightId is NULL')

    def load_orders(self, orders, departure, destination):
        availablePlanes = self.search_available_planes_for_airport(departure)
        route = f'{departure}-{destination}'
        
        if len(availablePlanes)>0:
            planeId = availablePlanes[0]
            
            prevId = self.dbManager.query_data('SELECT flightId FROM flights')[-1]
            newId = prevId + 1 # Note that this way of creating a new ID assumes that all flights are in chronological order (which must not be the case)
            
            newFlight = Flight(newId, route, 'scheduled')
            self.dbManager.insert_data('flights', newFlight.get_insert_columns(), newFlight.to_tuple()) # Create new flight
            self.dbManager.update_cell('cargo', 'flightId', newId, f'cargoId = {planeId}') # Assign plane to flight
            
            for order in orders:
                self.dbManager.update_cell('orders', 'flightId', newId, f'orderId = {order}') #Assign orders to flight
            
            return planeId
        
        else:
            return None #Will return None if there is no available plane

    '''Additional functions beyond the requirement of the assignment'''
    
    def set_arrival(self, flightId): 
        
        if flightId != None: 
            self.dbManager.update_cell('flights', 'status', '"archived"', f'flightId = {flightId}')
        
            aircraftId = self.dbManager.query_data(f'SELECT CargoId FROM cargo WHERE flightId = {flightId}')[0]
            arrivalAirport = self.dbManager.query_data(f'SELECT route FROM flights WHERE flightId = {flightId}')[0].split("-")[1]
        
            self.dbManager.update_cell('cargo', 'location', f'"{arrivalAirport}"', f'flightId = {flightId}')
            self.dbManager.update_cell('cargo', 'flightId', 'NULL', f'flightId = {flightId}')
        
            return [aircraftId, self.search_manifest(flightId)]
        
        else:
            return None
    
    def get_all_delivered_orders(self): # Returns a dictionary of all orders whose assigned flight is archived. The flightIds are the keys of the dictionary.
        query = self.dbManager.query_data('SELECT orderId, orders.flightId FROM orders INNER JOIN flights ON flights.flightId = orders.flightId WHERE status = "archived"')
        
        '''Algorithm to convert list of tuples into a dictionary:'''
        d = {}
        temp = 0
        for queryMatch in query:
            currVal = queryMatch[1]
            if currVal != temp: 
                d[currVal] = []
                d[currVal].append(queryMatch[0])
            else:
                d[currVal].append(queryMatch[0])
            temp = currVal
        
        return d
    
                
                
            
            
            
        
        
