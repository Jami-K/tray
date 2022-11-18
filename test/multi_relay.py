import hid

class Relay(object):
	def __init__(self, path=b'1-7:1.0'):
		self.h = hid.device()
		self.h.open_path(path)
		self.h.set_nonblocking(1)

	def get_switch_statuses_from_report(self, report):
		switch_statuses = report[7]
		switch_statuses = [int(x) for x in list('{0:08b}'.format(switch_statuses))]
		switch_statuses.reverse()

		return switch_statuses

	def send_feature_report(self, message):
		self.h.send_feature_report(message)

	def get_feature_report(self):
		feature = 1
		length = 8
		return self.h.get_feature_report(feature, length)

	def state(self, relay, on=None):
		# Getter
		if on == None:
			if relay == 0:
				report = self.get_feature_report()
				switch_statuses = self.get_switch_statuses_from_report(report)
				status = []
				for s in switch_statuses:
					status.append(bool(s))
			else:
				report = self.get_feature_report()
				switch_statuses = self.get_switch_statuses_from_report(report)
				status = bool(switch_statuses[relay-1])

			return status

		# Setter
		else:
			if relay == 0:
				if on:
					message = [0xFE]
				else:
					message = [0xFC]
			else:
				if on:
					message = [0xFF, relay]
				else:
					message = [0xFD, relay]

			self.send_feature_report(message)

if __name__ == '__main__':
	from time import sleep

	dic = []
	for i in hid.enumerate():
		print(i)
		if i['product_string'] == 'USBRelay2':
			dic.append(i['path'])
	print(dic)



	for j in range(3):
		relay1 = Relay(path=dic[0])
		relay2 = Relay(path=dic[1])
		# Create a relay object
		# (Setter) Turn switch 1 on
		relay1.state(0, on=True)
		relay2.state(0, on=True)

		relay1.h.close()
		relay2.h.close()

		# (Getter) Print the status of switch 1
		#print(relay1.state(1))

		sleep(0.3)

		relay1 = Relay(path=dic[0])
		relay2 = Relay(path=dic[1])
		# Turn all switches off
		relay1.state(0, on=False)
		relay2.state(0, on=False)

		# Print the state of all switches
		#print(relay1.state(0))
		sleep(0.3)

		relay1.h.close()
		relay2.h.close()










