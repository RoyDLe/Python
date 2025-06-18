import numpy
from DBObjects import Order, Flight, Aircraft
from DBManager import DBManager

numpy.random.seed()

class Generator():
    
    def __init__(self):
        
        self.dbManager = DBManager("cfv_start.db")  
    
    def generate_shipment_plan(self, flights, ordersPerFlight, unassignedOrders = 10):
        
        self.dbManager.create_table("flights", Flight.get_fields())
        self.dbManager.create_table("orders", Order.get_fields())
        
        flightList = []
        
        lowerBound = 1
        
        for flightId in range(1, flights + 1):
            
            route = self.random_route()
            
            flight = Flight(flightId, route, self.random_status())
            print(f'Flight {flight.flightId} with route {flight.route} created!')
            self.dbManager.insert_data("flights", flight.get_insert_columns(), flight.to_tuple())
            flightList.append(flight)
            
            for orderId in range(lowerBound, lowerBound + ordersPerFlight):
                order = Order(orderId, flightId, route.split("-")[0], route.split("-")[1], self.random_val(), self.random_val())
                print(f'{order.orderId}, {order.flightId}, {order.origin}, {order.destination}, {order.weight}, {order.volume}')
                self.dbManager.insert_data("orders", order.get_insert_columns(), order.to_tuple())
            lowerBound += ordersPerFlight
        
        upperBound = lowerBound + unassignedOrders
        
        while lowerBound < upperBound:
            route = self.random_route()
            order = Order(lowerBound,None,route.split("-")[0], route.split("-")[1], self.random_val(), self.random_val())
            self.dbManager.insert_data("orders", order.get_insert_columns(), order.to_tuple())
            lowerBound += 1
        
        return flightList
    
    def assign_flights(self, flightList, aircraftTypes, unassignedCargo = 5):
        
        self.dbManager.create_table("cargo", Aircraft.get_fields())
        
        cargoId = 1
        for flight in flightList:
            if flight.status == "scheduled":
                cargo = Aircraft(cargoId, flight.flightId, numpy.random.choice(aircraftTypes), flight.route.split("-")[0])
                self.dbManager.insert_data("cargo", cargo.get_insert_columns(), cargo.to_tuple())
                cargoId += 1
        
        for i in range(unassignedCargo):
            cargo = Aircraft(cargoId,None, numpy.random.choice(aircraftTypes), self.random_route().split("-")[0])
            self.dbManager.insert_data("cargo", cargo.get_insert_columns(), cargo.to_tuple())
            cargoId += 1
        
    def random_route(self):
        origin = numpy.random.choice(["MXP", "CDG", "MAD", "CGN", "LIN"])
        dest = numpy.random.choice(["LHR", "VIE", "EIN", "AMS", "BCN"])
        return f'{origin}-{dest}'
    
    def random_val(self):
        return numpy.random.random()*100
    
    def random_status(self):
        return numpy.random.choice(["scheduled", "archived"])
    
              


