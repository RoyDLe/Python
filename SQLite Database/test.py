from Generator import Generator
from assignment2 import CFV_Interface
from DBManager import DBManager


manager = DBManager("cfv_start.db") 
"""
cargoTypes = manager.query_data('SELECT PlaneType from CargoTypes')
gen = Generator()
plan = gen.generate_shipment_plan(10,5)
gen.assign_flights(plan, cargoTypes)
"""

cfv = CFV_Interface("cfv_start.db")
print(cfv.search_manifest(2))
print(cfv.search_flight_for_route("MXP", "VIE"))
print(cfv.search_unassigned_orders())
print(cfv.search_available_planes_for_airport("CGN"))
#print(cfv.load_orders([51], "CDG", "LHR"))

print(manager.query_data('SELECT origin, destination FROM orders WHERE orderId = 51'))
'''Note that the origin and destination of the flight does not need to match the origin and destination of the order assigned to it.
This is because the function load_orders does not check for consistency'''


#print(cfv.set_arrival(13))
print(cfv.get_all_delivered_orders())




