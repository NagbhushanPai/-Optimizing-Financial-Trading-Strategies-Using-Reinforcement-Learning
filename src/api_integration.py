from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def tickPrice(self, reqId, tickType, price, attrib):
        print(f"Tick Price. Ticker Id: {reqId}, Type: {tickType}, Price: {price}")

    def placeOrder(self, contract, order):
        order_id = self.nextOrderId()
        self.placeOrder(order_id, contract, order)
        return order_id

# Example usage
def connect_to_ib(host="127.0.0.1", port=7497, client_id=1):
    app = IBapi()
    app.connect(host, port, client_id)
    return app