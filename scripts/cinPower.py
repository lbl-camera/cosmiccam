import sys
from cosmic.camera.cin import CIN
c = CIN()

if len(sys.argv) < 2:
	print('Usage: python cinPower.py on(off)')
	sys.exit()

if sys.argv[1] == 'on':
	c.allPowerUp(testOnly = False)
	c.setExpTime(10)
elif sys.argv[1] == 'off':
	c.allPowerDown()
else: 
	print('Usage: python cinPower.py on(off)')
