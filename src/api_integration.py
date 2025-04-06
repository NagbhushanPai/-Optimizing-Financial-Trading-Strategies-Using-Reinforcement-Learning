from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import time

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []

    def tickPrice(self, reqId, tickType, price, attrib):
        print(f"Tick Price. Ticker Id: {reqId}, Type: {tickType}, Price: {price}")
        self.data.append((reqId, tickType, price))

    def placeOrder(self, contract, order, action="BUY", quantity=100):
        order_id = self.nextOrderId()
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        self.placeOrder(order_id, contract, order)
        return order_id

def connect_to_ib(host="127.0.0.1", port=7497, client_id=1):
    app = IBapi()
    app.connect(host, port, client_id)
    time.sleep(1)  # Allow connection to establish
    return app