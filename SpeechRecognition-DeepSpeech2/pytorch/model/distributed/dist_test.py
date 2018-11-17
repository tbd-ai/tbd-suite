import socket
import argparse
parser = argparse.ArgumentParser(description='Dist conn test')
parser.add_argument('--ip',default='127.0.0.1')
parser.add_argument('--port', type=int, default=5050)
args = parser.parse_args()

s = socket.socket()
address = args.ip
port = args.port  # port number is a number, not string
try:
    s.connect((address, port))
    print("sucess @ {}:{}".format(address,port))
    # originally, it was 
    # except Exception, e: 
    # but this syntax is not supported anymore. 
except Exception as e: 
    print("something's wrong with %s:%d. Exception is %s" % (address, port, e))
finally:
    s.close()
