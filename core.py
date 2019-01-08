import config
import math

def getInitialPrimitive():
    configData = config.getConfig()
    rho_inf = float(configData["core"]["rho_inf"])
    mach = float(configData["core"]["mach"])
    machcos = mach * math.cos(calculateTheta())
    machsin = mach * math.sin(calculateTheta())
    pr_inf = float(configData["core"]["pr_inf"])
    primal = [rho_inf, machcos, machsin, pr_inf]
    return primal

    
def calculateTheta():
    configData = config.getConfig()
    theta = math.radians(float(configData["core"]["aoa"]))
    return theta


