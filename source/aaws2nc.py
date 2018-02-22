import pandas as pd
import xarray as xr
from datetime import datetime
import pytz
from sunposition import sunpos
from common import write_data
import common

def aaws2nc(args, op_file, station_dict, station_name):

	freezing_point_temp = common.freezing_point_temp
	pascal_per_millibar = common.pascal_per_millibar
	seconds_in_hour = common.seconds_in_hour
	
	if args.fillvalue_float:
		fillvalue_float = args.fillvalue_float
	else:
		fillvalue_float = common.fillvalue_float
	
	header_rows = 8

	column_names = ['timestamp', 'air_temp', 'vtempdiff', 'rh', 'pressure', 'wind_dir', 'wind_spd']

	df = pd.read_csv(args.input_file or args.fl_in, skiprows = header_rows, skip_blank_lines=True, header=None, names = column_names)
	df.index.name = 'time'
	df.loc[:,'air_temp'] += freezing_point_temp
	df.loc[:,'pressure'] *= pascal_per_millibar
	df =  df.where((pd.notnull(df)), fillvalue_float)

	ds = xr.Dataset.from_dataframe(df)
	ds = ds.drop('time')


	# Intializing variables
	num_rows =  df['timestamp'].size
	time, time_bounds, sza = ([0]*num_rows for x in range(3))
	
	
	print('retrieving latitude and longitude...')
	
	f = open(args.input_file or args.fl_in)
	f.readline()
	for line in f:
		x = str(line[12:].strip('\n'))
		break
	f.close()

	if x == 'AGO-4':
		temp_stn = 'aaws_ago4'
	elif x == 'Alexander Tall Tower!':
		temp_stn = 'aaws_alexander'
	elif x == 'Austin':
		temp_stn = 'aaws_austin'
	elif x == 'Baldrick':
		temp_stn = 'aaws_baldrick'
	elif x == 'Bear Peninsula':
		temp_stn = 'aaws_bearpeninsula'
	elif x == 'Bonaparte Point':
		temp_stn = 'aaws_bonapartepoint'
	elif x == 'Byrd':
		temp_stn = 'aaws_byrd'
	elif x == 'Cape Bird':
		temp_stn = 'aaws_capebird'
	elif x == 'Cape Denison':
		temp_stn = 'aaws_capedenison'
	elif x == 'Cape Hallett':
		temp_stn = 'aaws_capehallett'
	elif x == 'D-10':
		temp_stn = 'aaws_d10'
	elif x == 'D-47':
		temp_stn = 'aaws_d47'
	elif x == 'D-85':
		temp_stn = 'aaws_d85'
	elif x == 'Dismal Island':
		temp_stn = 'aaws_dismalisland'
	elif x == 'Dome C II':
		temp_stn = 'aaws_domecII'
	elif x == 'Dome Fuji':
		temp_stn = 'aaws_domefuji'
	elif x == 'Elaine':
		temp_stn = 'aaws_elaine'
	elif x == 'Elizabeth':
		temp_stn = 'aaws_elizabeth'
	elif x == 'Emilia':
		temp_stn = 'aaws_emilia'
	elif x == 'Emma':
		temp_stn = 'aaws_emma'
	elif x == 'Erin':
		temp_stn = 'aaws_erin'
	elif x == 'Evans Knoll':
		temp_stn = 'aaws_evansknoll'
	elif x == 'Ferrell':
		temp_stn = 'aaws_ferrell'
	elif x == 'Gill':
		temp_stn = 'aaws_gill'
	elif x == 'Harry':
		temp_stn = 'aaws_harry'
	elif x == 'Henry':
		temp_stn = 'aaws_henry'
	elif x == 'Janet':
		temp_stn = 'aaws_janet'
	elif x == 'JASE2007':
		temp_stn = 'aaws_jase2007'
	elif x == 'Kathie':
		temp_stn = 'aaws_kathie'
	elif x == 'Kominko-Slade':
		temp_stn = 'aaws_kominkoslade'
	elif x == 'Laurie II':
		temp_stn = 'aaws_laurieII'
	elif x == 'Lettau':
		temp_stn = 'aaws_lettau'
	elif x == 'Linda':
		temp_stn = 'aaws_linda'
	elif x == 'Lorne':
		temp_stn = 'aaws_lorne'
	elif x == 'Manuela':
		temp_stn = 'aaws_manuela'
	elif x == 'Marble Point':
		temp_stn = 'aaws_marblepoint'
	elif x == 'Marble Point II':
		temp_stn = 'aaws_marblepointII'
	elif x == 'Margaret':
		temp_stn = 'aaws_margaret'
	elif x == 'Marilyn':
		temp_stn = 'aaws_marilyn'
	elif x == 'Minna Bluff':
		temp_stn = 'aaws_minnabluff'
	elif x == 'Mizuho':
		temp_stn = 'aaws_mizuho'
	elif x == 'Mount Siple':
		temp_stn = 'aaws_mountsiple'
	elif x == 'Nico':
		temp_stn = 'aaws_nico'
	elif x == 'PANDA-South':
		temp_stn = 'aaws_pandasouth'
	elif x == 'Pegasus North':
		temp_stn = 'aaws_pegasusnorth'
	elif x == 'Phoenix':
		temp_stn = 'aaws_phoenix'
	elif x == 'Port Martin':
		temp_stn = 'aaws_portmartin'
	elif x == 'Possession Island':
		temp_stn = 'aaws_possessionisland'
	elif x == 'Relay Station':
		temp_stn = 'aaws_relaystation'
	elif x == 'Sabrina':
		temp_stn = 'aaws_sabrina'
	elif x == 'Schwerdtfeger':
		temp_stn = 'aaws_schwerdtfeger'
	elif x == 'Siple Dome':
		temp_stn = 'aaws_sipledome'
	elif x == 'Theresa':
		temp_stn = 'aaws_theresa'
	elif x == 'Thurston Island':
		temp_stn = 'aaws_thurstonisland'
	elif x == 'Vito':
		temp_stn = 'aaws_vito'
	elif x == 'White Island':
		temp_stn = 'aaws_whiteisland'
	elif x == 'Whitlock':
		temp_stn = 'aaws_whitlock'
	elif x == 'Willie Field':
		temp_stn = 'aaws_williefield'
	elif x == 'Windless Bight':
		temp_stn = 'aaws_windlessbight'
	
	latitude = (station_dict.get(temp_stn)[0])
	longitude = (station_dict.get(temp_stn)[1])

	
	print('retrieving station name...')

	if args.station_name:
		print('Default station name overrided by user provided station name')
	else:
		station_name = x


	print('calculating time and sza...')
	
	tz = pytz.timezone(args.timezone)
	dtime_1970 = datetime(1970,1,1)
	dtime_1970 = tz.localize(dtime_1970.replace(tzinfo=None))
	i = 0
	
	with open(args.input_file or args.fl_in, "r") as infile:
		for line in infile.readlines()[header_rows:]:
			temp_dtime = datetime.strptime(line.strip().split(",")[0], '%Y-%m-%dT%H:%M:%SZ')
			temp_dtime = tz.localize(temp_dtime.replace(tzinfo=None))		
			time[i] = (temp_dtime-dtime_1970).total_seconds()
			
			time_bounds[i] = (time[i]-seconds_in_hour, time[i])
			
			sza[i] = sunpos(temp_dtime,latitude,longitude,0)[1]
			
			i += 1

	ds['time'] = (('time'),time)
	ds['time_bounds'] = (('time', 'nbnd'),time_bounds)
	ds['sza'] = (('time'),sza)
	ds['station_name'] = ((),station_name)
	ds['latitude'] = ((),latitude)
	ds['longitude'] = ((),longitude)
	

	ds.attrs = {'source':'surface observation', 'featureType':'timeSeries', 'institution':'UW SSEC', 'reference':'https://amrc.ssec.wisc.edu/', 'Conventions':'CF-1.7', 'data_type':'q1h', 'time_convention':"'time: point' variables match the time coordinate values exactly, whereas 'time: mean' variables are valid for the mean time within the time_bounds variable." + " e.g.: air_temp is continuously measured and then hourly-mean values are stored for each period contained in the time_bounds variable"}

	ds['air_temp'].attrs= {'units':'kelvin', 'long_name':'air temperature', 'standard_name':'air_temperature', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['vtempdiff'].attrs= {'units':'1', 'long_name':'vertical temperature differential', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['rh'].attrs= {'units':'1', 'long_name':'relative humidity', 'standard_name':'relative_humidity', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['pressure'].attrs= {'units':'pascal', 'long_name':'air pressure', 'standard_name':'air_pressure', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['wind_dir'].attrs= {'units':'degree', 'long_name':'wind direction', 'standard_name':'wind_from_direction', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['wind_spd'].attrs= {'units':'meter second-1', 'long_name':'wind speed', 'standard_name':'wind_speed', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['time'].attrs= {'units':'seconds since 1970-01-01 00:00:00', 'long_name':'time of measurement',	'standard_name':'time', 'bounds':'time_bounds', 'calendar':'noleap'}
	ds['sza'].attrs= {'units':'degree', 'long_name':'Solar Zenith Angle', 'standard_name':'solar_zenith_angle', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['station_name'].attrs= {'long_name':'Station Name', 'cf_role':'timeseries_id'}
	ds['latitude'].attrs= {'units':'degrees_north', 'standard_name':'latitude'}
	ds['longitude'].attrs= {'units':'degrees_east', 'standard_name':'longitude'}
	

	encoding = {'air_temp': {'_FillValue': fillvalue_float, 'dtype': 'f4'},
				'vtempdiff': {'_FillValue': fillvalue_float, 'dtype': 'f4'},
				'rh': {'_FillValue': fillvalue_float, 'dtype': 'f4'},
				'pressure': {'_FillValue': fillvalue_float, 'dtype': 'f4'},
				'wind_dir': {'_FillValue': fillvalue_float, 'dtype': 'f4'},
				'wind_spd': {'_FillValue': fillvalue_float, 'dtype': 'f4'},
				'time': {'_FillValue': False},
				'time_bounds': {'_FillValue': False},
				'sza': {'_FillValue': False},
				'latitude': {'_FillValue': False},
				'longitude': {'_FillValue': False}
				}


	write_data(args, ds, op_file, encoding)