import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pytz import timezone
from sunposition import sunpos
from common import write_data

def gcnet2nc(args, op_file, station_dict, station_name, convert_temp, convert_press, seconds_in_hour, fillvalue_double):

	header_lines = 54
	check_na = 999.0
	hour_conversion = (100/4)		#Divided by 4 because each hour value is a multiple of 4 and then multiplied by 100 to convert decimal to integer
	last_hour = 23

	column_names = ['station_number', 'year', 'julian_decimal_time', 'sw_down', 'sw_up', 'net_radiation', 'temperature_tc_1', 'temperature_tc_2', 'temperature_cs500_1', 'temperature_cs500_2', 'relative_humidity_1', 'relative_humidity_2', 
	'u1_wind_speed', 'u2_wind_speed', 'u_direction_1', 'u_direction_2', 'atmos_pressure', 'snow_height_1', 'snow_height_2', 't_snow_01', 't_snow_02', 't_snow_03', 't_snow_04', 't_snow_05', 't_snow_06', 't_snow_07', 't_snow_08', 't_snow_09', 't_snow_10', 
	'battery_voltage', 'sw_down_max', 'sw_up_max', 'net_radiation_max', 'max_air_temperature_1', 'max_air_temperature_2', 'min_air_temperature_1', 'min_air_temperature_2', 'max_windspeed_u1', 'max_windspeed_u2', 'stdev_windspeed_u1', 'stdev_windspeed_u2', 
	'ref_temperature', 'windspeed_2m', 'windspeed_10m', 'wind_sensor_height_1', 'wind_sensor_height_2', 'albedo', 'zenith_angle', 'qc1', 'qc9', 'qc17', 'qc25']

	
	df = pd.read_csv(args.input_file or args.fl_in, delim_whitespace=True, skiprows=header_lines, skip_blank_lines=True, header=None, names = column_names)
	df.index.name = 'time'
	df['qc25'] = df['qc25'].astype(str)			# To avoid 999 values marked as N/A
	df.replace(check_na, np.nan, inplace=True)
	df.loc[:,['temperature_tc_1', 'temperature_tc_2', 'temperature_cs500_1', 'temperature_cs500_2', 't_snow_01', 't_snow_02', 't_snow_03', 't_snow_04', 't_snow_05', 't_snow_06', 't_snow_07', 't_snow_08', 't_snow_09', 't_snow_10', 'max_air_temperature_1', 'max_air_temperature_2', 'min_air_temperature_1', 'min_air_temperature_2', 'ref_temperature']] += convert_temp
	df.loc[:,'atmos_pressure'] *= convert_press
	df = df.where((pd.notnull(df)), fillvalue_double)
	df['qc25'] = df['qc25'].astype(int)			#Convert it back to int

	station_number = df['station_number'][0]
	df.drop('station_number', axis=1, inplace=True)

	ds = xr.Dataset.from_dataframe(df)
	ds = ds.drop('time')
	
	
	# Intializing variables
	num_lines =  df['year'].size
	qc_swdn, qc_swup, qc_netradiation, qc_ttc1, qc_ttc2, qc_tcs1, qc_tcs2, qc_rh1, qc_rh2, qc_u1, qc_u2, qc_ud1, qc_ud2, qc_pressure, qc_snowheight1, qc_snowheight2, qc_tsnow1, qc_tsnow2, qc_tsnow3, qc_tsnow4, qc_tsnow5, qc_tsnow6, qc_tsnow7, qc_tsnow8, qc_tsnow9, qc_tsnow10, qc_battery = ([0]*num_lines for x in range(27))

	hour, month, day, time, time_bounds, sza = ([0]*num_lines for x in range(6))

	print('calculating quality control variables...')
	temp1 = [list(map(int, i)) for i in zip(*map(str, df['qc1']))]
	temp9 = [list(map(int, i)) for i in zip(*map(str, df['qc9']))]
	temp17 = [list(map(int, i)) for i in zip(*map(str, df['qc17']))]
	temp25 = [list(map(int, i)) for i in zip(*map(str, df['qc25']))]

	
	qc_swdn[:] = temp1[0]
	qc_swup[:] = temp1[1]
	qc_netradiation[:] = temp1[2]
	qc_ttc1[:] = temp1[3]
	qc_ttc2[:] = temp1[4]
	qc_tcs1[:] = temp1[5]
	qc_tcs2[:] = temp1[6]
	qc_rh1[:] = temp1[7]
	
	qc_rh2[:] = temp9[0]
	qc_u1[:] = temp9[1]
	qc_u2[:] = temp9[2]
	qc_ud1[:] = temp9[3]
	qc_ud2[:] = temp9[4]
	qc_pressure[:] = temp9[5]
	qc_snowheight1[:] = temp9[6]
	qc_snowheight2[:] = temp9[7]

	qc_tsnow1[:] = temp17[0]
	qc_tsnow2[:] = temp17[1]
	qc_tsnow3[:] = temp17[2]
	qc_tsnow4[:] = temp17[3]
	qc_tsnow5[:] = temp17[4]
	qc_tsnow6[:] = temp17[5]
	qc_tsnow7[:] = temp17[6]
	qc_tsnow8[:] = temp17[7]

	qc_tsnow9[:] = temp25[0]
	qc_tsnow10[:] = temp25[1]
	qc_battery[:] = temp25[2]

	ds['qc_swdn'] = (('time',qc_swdn))
	ds['qc_swup'] = (('time',qc_swup))
	ds['qc_netradiation'] = (('time',qc_netradiation))
	ds['qc_ttc1'] = (('time',qc_ttc1))
	ds['qc_ttc2'] = (('time',qc_ttc2))
	ds['qc_tcs1'] = (('time',qc_tcs1))
	ds['qc_tcs2'] = (('time',qc_tcs2))
	ds['qc_rh1'] = (('time',qc_rh1))
	ds['qc_rh2'] = (('time',qc_rh2))
	ds['qc_u1'] = (('time',qc_u1))
	ds['qc_u2'] = (('time',qc_u2))
	ds['qc_ud1'] = (('time',qc_ud1))
	ds['qc_ud2'] = (('time',qc_ud2))
	ds['qc_pressure'] = (('time',qc_pressure))
	ds['qc_snowheight1'] = (('time',qc_snowheight1))
	ds['qc_snowheight2'] = (('time',qc_snowheight2))
	ds['qc_tsnow1'] = (('time',qc_tsnow1))
	ds['qc_tsnow2'] = (('time',qc_tsnow2))
	ds['qc_tsnow3'] = (('time',qc_tsnow3))
	ds['qc_tsnow4'] = (('time',qc_tsnow4))
	ds['qc_tsnow5'] = (('time',qc_tsnow5))
	ds['qc_tsnow6'] = (('time',qc_tsnow6))
	ds['qc_tsnow7'] = (('time',qc_tsnow7))
	ds['qc_tsnow8'] = (('time',qc_tsnow8))
	ds['qc_tsnow9'] = (('time',qc_tsnow9))
	ds['qc_tsnow10'] = (('time',qc_tsnow10))
	ds['qc_battery'] = (('time',qc_battery))

	
	print('retrieving lat and lon...')
	if station_number == 1:
		temp_stn = 'gcnet_swiss'
	elif station_number == 2:
		temp_stn = 'gcnet_crawford'
	elif station_number == 3:
		temp_stn = 'gcnet_nasa-u'
	elif station_number == 4:
		temp_stn = 'gcnet_gits'
	elif station_number == 5:
		temp_stn = 'gcnet_humboldt'
	elif station_number == 6:
		temp_stn = 'gcnet_summit'
	elif station_number == 7:
		temp_stn = 'gcnet_tunu-n'
	elif station_number == 8:
		temp_stn = 'gcnet_dye2'
	elif station_number == 9:
		temp_stn = 'gcnet_jar'
	elif station_number == 10:
		temp_stn = 'gcnet_saddle'
	elif station_number == 11:
		temp_stn = 'gcnet_dome'
	elif station_number == 12:
		temp_stn = 'gcnet_nasa-e'
	elif station_number == 13:
		temp_stn = 'gcnet_cp2'
	elif station_number == 14:
		temp_stn = 'gcnet_ngrip'
	elif station_number == 15:
		temp_stn = 'gcnet_nasa-se'
	elif station_number == 16:
		temp_stn = 'gcnet_kar'
	elif station_number == 17:
		temp_stn = 'gcnet_jar2'
	elif station_number == 18:
		temp_stn = 'gcnet_kulu'
	elif station_number == 19:
		temp_stn = 'gcnet_jar3'
	elif station_number == 20:
		temp_stn = 'gcnet_aurora'
	elif station_number == 21 or 26:
		temp_stn = 'gcnet_petermann-gl'
	elif station_number == 22:
		temp_stn = 'gcnet_peterman-ela'
	elif station_number == 23:
		temp_stn = 'gcnet_neem'
	elif station_number == 30:
		temp_stn = 'gcnet_lar1'
	elif station_number == 31:
		temp_stn = 'gcnet_lar2'
	elif station_number == 32:
		temp_stn = 'gcnet_lar3'
	
	latitude = station_dict.get(temp_stn)[0]
	longitude = station_dict.get(temp_stn)[1]

	if args.station_name:
		print('Default station name overrided by user provided station name')
	else:
		station_name = station_dict.get(temp_stn)[2]

	

	
	print('calculating hour...')
	hour[:] = [int(x) for x in [round((v-int(v)),3)*hour_conversion for v in df['julian_decimal_time']]]
	
	i = 0
	while i < num_lines:
		if hour[i] > last_hour:
			hour[i] = 0
		i += 1

	print("calculating time and sza...")
	
	j = 0
	while j < num_lines:
		dt = datetime.strptime("%s %s %s" % (df['year'][j], int(df['julian_decimal_time'][j]), hour[j]), "%Y %j %H").replace(tzinfo = timezone(args.timezone))		
		time[j] = (dt-(datetime(1970,1,1)).replace(tzinfo = timezone(args.timezone))).total_seconds()
		time_bounds[j] = (time[j]-seconds_in_hour, time[j])
		
		sza[j] = sunpos(dt,latitude,longitude,0)[1]
		j += 1
	

	if args.analysis:
		print('calculating month and day...')
		def get_month_day(year, day, one_based=False):
			if one_based:  # if Jan 1st is 1 instead of 0
				day -= 1
			dt = datetime(year, 1, 1) + timedelta(days=day)
			return dt.month, dt.day

		j = 0
		while j < num_lines:
			month[j] = get_month_day(int(df['year'][j]), int(df['julian_decimal_time'][j]), True)[0]
			day[j] = get_month_day(int(df['year'][j]), int(df['julian_decimal_time'][j]), True)[1]
			j += 1


		ds['hour'] = (('time'),hour)
		ds['month'] = (('time'),month)
		ds['day'] = (('time'),day)


	ds['time'] = (('time'),time)
	ds['time_bounds'] = (('time', 'nbnd'),time_bounds)
	ds['sza'] = (('time'),sza)
	ds['station_number'] = ((),station_number)
	ds['station_name'] = ((),station_name)
	ds['latitude'] = ((),latitude)
	ds['longitude'] = ((),longitude)

	ds.attrs = {'title':'Surface Radiation Data from Greenland Climate Network', 'source':'Surface Observations', 'featureType':'timeSeries', 'institution':'Cooperative Institute for Research in Enviornmental Sciences', 
	'reference':'http://cires.colorado.edu/science/groups/steffen/gcnet/', 'Conventions':'CF-1.7', 'time_convention':"'time: point' variables match the time coordinate values exactly, whereas 'time: mean' variables are valid for the mean time within the time_bounds variable." + " e.g.: battery_voltage is measured once per hour at the time stored in the 'time' coordinate." + 	" On the other hand, temperature_tc_1 is continuously measured and then hourly-mean values are stored for each period contained in the time_bounds variable"}

	ds['station_number'].attrs= {'units':'1', 'long_name':'Station Number'}
	ds['year'].attrs= {'units':'1', 'long_name':'Year'}
	ds['julian_decimal_time'].attrs= {'units':'decimal time', 'long_name':'Julian Decimal Time', 'note':'For each year, time starts at 1.0000 and ends at 365.9999.'}
	ds['sw_down'].attrs= {'units':'watt meter-2', 'long_name':'Shortwave Flux Down', 'standard_name':'downwelling_shortwave_flux_in_air', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['sw_up'].attrs= {'units':'watt meter-2', 'long_name':'Shortwave Flux Up', 'standard_name':'upwelling_shortwave_flux_in_air', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['net_radiation'].attrs= {'units':'watt meter-2', 'long_name':'Net Radiation', 'standard_name':'surface_net_downward_radiative_flux', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['temperature_tc_1'].attrs= {'units':'kelvin', 'long_name':'TC-1 Air Temperature', 'standard_name':'air_temperature', 'note':'air temperature from TC sensor', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['temperature_tc_2'].attrs= {'units':'kelvin', 'long_name':'TC-2 Air Temperature', 'standard_name':'air_temperature', 'note':'air temperature from TC sensor', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['temperature_cs500_1'].attrs= {'units':'kelvin', 'long_name':'CS500-1 Air Temperature', 'standard_name':'air_temperature', 'note':'air temperature from CS500 sensor', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['temperature_cs500_2'].attrs= {'units':'kelvin', 'long_name':'CS500-2 Air Temperature', 'standard_name':'air_temperature', 'note':'air temperature from CS500 sensor', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['relative_humidity_1'].attrs= {'units':'1', 'long_name':'Relative Humidity 1', 'standard_name':'realtive_humidity', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['relative_humidity_2'].attrs= {'units':'1', 'long_name':'Relative Humidity 2', 'standard_name':'realtive_humidity', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['u1_wind_speed'].attrs= {'units':'meter second-1', 'long_name':'U1 Wind Speed', 'standard_name':'wind_speed', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['u2_wind_speed'].attrs= {'units':'meter second-1', 'long_name':'U2 Wind Speed', 'standard_name':'wind_speed', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['u_direction_1'].attrs= {'units':'degree', 'long_name':'U Direction 1', 'standard_name':'wind_from_direction', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['u_direction_2'].attrs= {'units':'degree', 'long_name':'U Direction 2', 'standard_name':'wind_from_direction', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['atmos_pressure'].attrs= {'units':'pascal', 'long_name':'Atmospheric Pressure', 'standard_name':'surface_air_pressure', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['snow_height_1'].attrs= {'units':'meter', 'long_name':'Snow Height 1', 'standard_name':'snow_height', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['snow_height_2'].attrs= {'units':'meter', 'long_name':'Snow Height 2', 'standard_name':'snow_height', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_01'].attrs= {'units':'kelvin', 'long_name':'T Snow 1', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_02'].attrs= {'units':'kelvin', 'long_name':'T Snow 2', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_03'].attrs= {'units':'kelvin', 'long_name':'T Snow 3', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_04'].attrs= {'units':'kelvin', 'long_name':'T Snow 4', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_05'].attrs= {'units':'kelvin', 'long_name':'T Snow 5', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_06'].attrs= {'units':'kelvin', 'long_name':'T Snow 6', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_07'].attrs= {'units':'kelvin', 'long_name':'T Snow 7', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_08'].attrs= {'units':'kelvin', 'long_name':'T Snow 8', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_09'].attrs= {'units':'kelvin', 'long_name':'T Snow 9', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['t_snow_10'].attrs= {'units':'kelvin', 'long_name':'T Snow 10', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['battery_voltage'].attrs= {'units':'volts', 'long_name':'Battery Voltage', 'standard_name':'battery_voltage', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['sw_down_max'].attrs= {'units':'watt meter-2', 'long_name':'Shortwave Flux Down Max', 'standard_name':'maximum_downwelling_shortwave_flux_in_air', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['sw_up_max'].attrs= {'units':'watt meter-2', 'long_name':'Shortwave Flux Up Max', 'standard_name':'maximum_upwelling_shortwave_flux_in_air', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['net_radiation_max'].attrs= {'units':'watt meter-2', 'long_name':'Net Radiation Max', 'standard_name':'maximum_net_radiation', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['max_air_temperature_1'].attrs= {'units':'kelvin', 'long_name':'Max Air Temperature 1', 'standard_name':'air_temperature', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['max_air_temperature_2'].attrs= {'units':'kelvin', 'long_name':'Max Air Temperature 2', 'standard_name':'air_temperature', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['min_air_temperature_1'].attrs= {'units':'kelvin', 'long_name':'Min Air Temperature 1', 'standard_name':'air_temperature', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['min_air_temperature_2'].attrs= {'units':'kelvin', 'long_name':'Min Air Temperature 2', 'standard_name':'air_temperature', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['max_windspeed_u1'].attrs= {'units':'meter second-1', 'long_name':'Max Windspeed-U1', 'standard_name':'wind_speed', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['max_windspeed_u2'].attrs= {'units':'meter second-1', 'long_name':'Max Windspeed-U2', 'standard_name':'wind_speed', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['stdev_windspeed_u1'].attrs= {'units':'meter second-1', 'long_name':'StdDev Windspeed-U1', 'standard_name':'wind_speed', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['stdev_windspeed_u2'].attrs= {'units':'meter second-1', 'long_name':'StdDev Windspeed-U2', 'standard_name':'wind_speed', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['ref_temperature'].attrs= {'units':'kelvin', 'long_name':'Reference Temperature', 'standard_name':'Need to ask network manager about long name', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['windspeed_2m'].attrs= {'units':'meter second-1', 'long_name':'Windspeed@2m', 'standard_name':'wind_speed', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['windspeed_10m'].attrs= {'units':'meter second-1', 'long_name':'Windspeed@10m', 'standard_name':'wind_speed', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['wind_sensor_height_1'].attrs= {'units':'meter', 'long_name':'Wind Sensor Height 1', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['wind_sensor_height_2'].attrs= {'units':'meter', 'long_name':'Wind Sensor Height 2', 'standard_name':'', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['albedo'].attrs= {'units':'1', 'long_name':'Albedo', 'standard_name':'surface_albedo', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['zenith_angle'].attrs= {'units':'degree', 'long_name':'Zenith Angle', 'standard_name':'solar_zenith_angle', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['qc1'].attrs= {'units':'1', 'long_name':'Quality Control variables 01-08', 'coordinates':'longitude latitude'}
	ds['qc9'].attrs= {'units':'1', 'long_name':'Quality Control variables 09-16', 'coordinates':'longitude latitude'}
	ds['qc17'].attrs= {'units':'1', 'long_name':'Quality Control variables 17-24', 'coordinates':'longitude latitude'}
	ds['qc25'].attrs= {'units':'1', 'long_name':'Quality Control variables 25-27', 'coordinates':'longitude latitude'}
	ds['qc_swdn'].attrs= {'units':'1', 'long_name':'Quality Control flag for Shortwave Flux Down'}
	ds['qc_swup'].attrs= {'units':'1', 'long_name':'Quality Control flag for Shortwave Flux Up'}
	ds['qc_netradiation'].attrs= {'units':'1', 'long_name':'Quality Control flag for Net Radiation'}
	ds['qc_ttc1'].attrs= {'units':'1', 'long_name':'Quality Control flag for TC-1 Air Temperature'}
	ds['qc_ttc2'].attrs= {'units':'1', 'long_name':'Quality Control flag for TC-2 Air Temperature'}
	ds['qc_tcs1'].attrs= {'units':'1', 'long_name':'Quality Control flag for CS500-1 Air Temperature'}
	ds['qc_tcs2'].attrs= {'units':'1', 'long_name':'Quality Control flag for CS500-2 Air Temperature'}
	ds['qc_rh1'].attrs= {'units':'1', 'long_name':'Quality Control flag for Relative Humidity 1'}
	ds['qc_rh2'].attrs= {'units':'1', 'long_name':'Quality Control flag for Relative Humidity 2'}
	ds['qc_u1'].attrs= {'units':'1', 'long_name':'Quality Control flag for U1 Wind Speed'}
	ds['qc_u2'].attrs= {'units':'1', 'long_name':'Quality Control flag for U2 Wind Speed'}
	ds['qc_ud1'].attrs= {'units':'1', 'long_name':'Quality Control flag for U Direction 1'}
	ds['qc_ud2'].attrs= {'units':'1', 'long_name':'Quality Control flag for U Direction 2'}
	ds['qc_pressure'].attrs= {'units':'1', 'long_name':'Quality Control flag for Atmospheric Pressure'}
	ds['qc_snowheight1'].attrs= {'units':'1', 'long_name':'Quality Control flag for Snow Height 1'}
	ds['qc_snowheight2'].attrs= {'units':'1', 'long_name':'Quality Control flag for Snow Height 2'}
	ds['qc_tsnow1'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 1'}
	ds['qc_tsnow2'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 2'}
	ds['qc_tsnow3'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 3'}
	ds['qc_tsnow4'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 4'}
	ds['qc_tsnow5'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 5'}
	ds['qc_tsnow6'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 6'}
	ds['qc_tsnow7'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 7'}
	ds['qc_tsnow8'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 8'}
	ds['qc_tsnow9'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 9'}
	ds['qc_tsnow10'].attrs= {'units':'1', 'long_name':'Quality Control flag for T Snow 10'}
	ds['qc_battery'].attrs= {'units':'1', 'long_name':'Quality Control flag for Battery Voltage'}
	ds['time'].attrs= {'units':'seconds since 1970-01-01 00:00:00', 'long_name':'time of measurement',	'standard_name':'time', 'bounds':'time_bounds', 'calendar':'noleap'}
	ds['sza'].attrs= {'units':'degree', 'long_name':'Solar Zenith Angle', 'standard_name':'solar_zenith_angle', 'coordinates':'longitude latitude', 'cell_methods':'time: mean'}
	ds['station_name'].attrs= {'long_name':'Station Name', 'cf_role':'timeseries_id'}
	ds['latitude'].attrs= {'units':'degrees_north', 'standard_name':'latitude'}
	ds['longitude'].attrs= {'units':'degrees_east', 'standard_name':'longitude'}
	

	encoding = {'julian_decimal_time': {'_FillValue': False},
				'sw_down': {'_FillValue': fillvalue_double},
				'sw_up': {'_FillValue': fillvalue_double},
				'net_radiation': {'_FillValue': fillvalue_double},
				'temperature_tc_1': {'_FillValue': fillvalue_double},
				'temperature_tc_2': {'_FillValue': fillvalue_double},
				'temperature_cs500_1': {'_FillValue': fillvalue_double},
				'temperature_cs500_2': {'_FillValue': fillvalue_double},
				'relative_humidity_1': {'_FillValue': fillvalue_double},
				'relative_humidity_2': {'_FillValue': fillvalue_double},
				'u1_wind_speed': {'_FillValue': fillvalue_double},
				'u2_wind_speed': {'_FillValue': fillvalue_double},
				'u_direction_1': {'_FillValue': fillvalue_double},
				'u_direction_2': {'_FillValue': fillvalue_double},
				'atmos_pressure': {'_FillValue': fillvalue_double},
				'snow_height_1': {'_FillValue': fillvalue_double},
				'snow_height_2': {'_FillValue': fillvalue_double},
				't_snow_01': {'_FillValue': fillvalue_double},
				't_snow_02': {'_FillValue': fillvalue_double},
				't_snow_03': {'_FillValue': fillvalue_double},
				't_snow_04': {'_FillValue': fillvalue_double},
				't_snow_05': {'_FillValue': fillvalue_double},
				't_snow_06': {'_FillValue': fillvalue_double},
				't_snow_07': {'_FillValue': fillvalue_double},
				't_snow_08': {'_FillValue': fillvalue_double},
				't_snow_09': {'_FillValue': fillvalue_double},
				't_snow_10': {'_FillValue': fillvalue_double},
				'battery_voltage': {'_FillValue': fillvalue_double},
				'sw_down_max': {'_FillValue': fillvalue_double},
				'sw_up_max': {'_FillValue': fillvalue_double},
				'net_radiation_max': {'_FillValue': fillvalue_double},
				'max_air_temperature_1': {'_FillValue': fillvalue_double},
				'max_air_temperature_2': {'_FillValue': fillvalue_double},
				'min_air_temperature_1': {'_FillValue': fillvalue_double},
				'min_air_temperature_2': {'_FillValue': fillvalue_double},
				'max_windspeed_u1': {'_FillValue': fillvalue_double},
				'max_windspeed_u2': {'_FillValue': fillvalue_double},
				'stdev_windspeed_u1': {'_FillValue': fillvalue_double},
				'stdev_windspeed_u2': {'_FillValue': fillvalue_double},
				'ref_temperature': {'_FillValue': fillvalue_double},
				'windspeed_2m': {'_FillValue': fillvalue_double},
				'windspeed_10m': {'_FillValue': fillvalue_double},
				'wind_sensor_height_1': {'_FillValue': fillvalue_double},
				'wind_sensor_height_2': {'_FillValue': fillvalue_double},
				'albedo': {'_FillValue': fillvalue_double},
				'zenith_angle': {'_FillValue': fillvalue_double},
				'time': {'_FillValue': False},
				'time_bounds': {'_FillValue': False},
				'sza': {'_FillValue': False},
				'latitude': {'_FillValue': False},
				'longitude': {'_FillValue': False}
				}


	write_data(args, ds, op_file, encoding)