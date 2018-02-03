from sunposition import sunpos
import gcnet2nc, promice2nc, aaws2nc

def time_calc(year,month,day,hour):
	delta = datetime(year,month,day,hour)-datetime(1970, 1, 1)
	return delta.total_seconds()

def solar(year,month,day,hour,lat,lon):
	temp_datetime = datetime(year,month,day,hour)
	return sunpos(temp_datetime,lat,lon,0)[1]
