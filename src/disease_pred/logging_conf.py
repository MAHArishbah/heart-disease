import logging
import os
import sys


try:
    from pythonjsonlogger.json import JsonFormatter
except ImportError:
    from pythonjsonlogger.jsonlogger import JsonFormatter

class CloudLoggingFormatter(JsonFormatter):
    def add_fields(self,log_record,record,message_dict): 
        super().add_fields(log_record,record,message_dict) #hook to add fields to every line
        log_record["severity"]=record.levelname
        log_record["logger"]=record.name
        log_record["service"]=os.getenv("K_SERVICE","local") #• K_SERVICE is set by Cloud Run to the service name, in local its local . same name for cloud run and local, without if
        log_record["model_version"]=os.getenv("MODEL_VERSION","dev")

def configure_logging()->None:
    handler=logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingFormatter("%(asctime)s %(message)s"))

    root=logging.getLogger()
    root.handlers=[handler]
    root.setLevel(os.getenv("LOG_LEVEL","INFO"))

    #uvicorn installs its own handlers, routing them through here
    for name in ("uvicorn","uvicorn.error"):
        logging.getLogger(name).handlers=[handler]
        logging.getLogger(name).propagate=False
    logging.getLogger("uvicorn.access").disabled=True #disables the uvicorn access logs 

