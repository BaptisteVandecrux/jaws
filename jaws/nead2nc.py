import numpy as np
import pandas as pd
import xarray as xr
import sys
import nead

try:
    from metpy.units import units
    from metpy.calc import potential_temperature, density
    from metpy.calc import mixing_ratio_from_relative_humidity, specific_humidity_from_mixing_ratio
except ImportError:
    print('ImportError: No module named metpy.units')
    print('HINT: This is a known issue for a dependent package that occurs using pip installation in Python 2.7 \n'
          'To fix it, please perform either of below operations: \n'
          '1. Install jaws using conda as: conda install -c conda-forge jaws \n'
          '2. Upgrade to Python version >= 3.6 \n')
    print('If none of the above works for you, '
          'please inform the JAWS maintainers by opening an issue at https://github.com/jaws/jaws/issues.')

    sys.exit(1)

try:
    from jaws import common, sunposition, clearsky, tilt_angle, fsds_adjust
except ImportError:
    import common, sunposition, clearsky, tilt_angle, fsds_adjust


def init_dataframe(args, input_file):
    """Initialize dataframe with data from input file; convert temperature and pressure to SI units"""
    df, columns = common.load_dataframe('nead', input_file, 0)

    temperature_vars = [
        'ta_tc1', 'ta_tc2', 'ta_cs1', 'ta_cs2',
        'tsn1', 'tsn2', 'tsn3','tsn4', 'tsn5',
        'tsn6', 'tsn7', 'tsn8', 'tsn9', 'tsn10',
        'ta_max1', 'ta_max2', 'ta_min1','ta_min2', 'ref_temp']

    temperature_vars = [e for e in temperature_vars if e in df.columns.values]
                                        
    if not args.celsius:
        df.loc[:, temperature_vars] += common.freezing_point_temp  # Convert units to Kelvin

    pressure_vars = ['ps']
    if not args.mb:
        df.loc[:, pressure_vars] *= common.pascal_per_millibar  # Convert units to millibar/hPa

    df = df.where((pd.notnull(df)), common.get_fillvalue(args))

    return df, temperature_vars, pressure_vars


def get_station(args, input_file, stations):
    """Get latitude, longitude and name for each station"""
    ds = nead.read(input_file)

    if 'station_num' in ds.attrs:
        station_number = ds.attrs['station_num']
    
        if 1 <= station_number <= 23:
            station = list(stations.values())[station_number]
        elif 30 <= station_number <= 32:
            name = 'gcnet_lar{}'.format(station_number - 29)
            station = stations[name]
        else:
            print('KeyError: {}'.format(ds.attrs['station_id']))
            print('HINT: This KeyError can occur when JAWS is asked to process station that is not in its database. '
                  'Please inform the JAWS maintainers by opening an issue at https://github.com/jaws/jaws/issues.')
            sys.exit(1)
    
        lat, lon, stn_nm = common.parse_station(args, station)
    else:
        coord = [float(f) for f in ds.attrs['geometry'].split('(')[1].split(')')[0].split(' ')]
        lat = coord[1]
        lon = coord[0]
        stn_nm =   ds.attrs['station_name']      
    return lat, lon, stn_nm
                
def get_time_and_sza(args, dataframe, longitude, latitude):
    """Calculate additional time related variables"""

    hour = dataframe['hour']
    
    dtime_1970, tz = common.time_common(args.tz)
    num_rows = dataframe['year'].size
    sza, az = ([0] * num_rows for _ in range(2))

    dataframe['dtime'] = pd.to_datetime(dataframe.timestamp.values)
    # Each timestamp is average of previous and current hour values i.e. value at hour=5 is average of hour=4 and hour=5
    # Our 'time' variable will represent values at half-hour i.e. 4.5 in above case, so subtract 30 minutes from all.
    dataframe['dtime'] -= pd.to_timedelta(common.seconds_in_half_hour, unit='s')

    dataframe['dtime'] = [tz.localize(i.replace(tzinfo=None)) for i in dataframe['dtime']]  # Set timezone

    time = (dataframe['dtime'] - dtime_1970) / np.timedelta64(1, 's')  # Seconds since 1970
    time_bounds = [(i-common.seconds_in_half_hour, i+common.seconds_in_half_hour) for i in time]
    month = pd.DatetimeIndex(dataframe['dtime']).month.values
    day = pd.DatetimeIndex(dataframe['dtime']).day.values
    minutes = pd.DatetimeIndex(dataframe['dtime']).minute.values
    dates = list(pd.DatetimeIndex(dataframe['dtime']).date)
    dates = [int(d.strftime("%Y%m%d")) for d in dates]
    first_date = min(dates)
    last_date = max(dates)

    for idx in range(num_rows):
        solar_angles = sunposition.sunpos(dataframe['dtime'][idx], latitude, longitude, 0)
        az[idx] = solar_angles[0]
        sza[idx] = solar_angles[1]

    return month, day, hour, minutes, time, time_bounds, sza, az, first_date, last_date


# Just a framework, need to do calculations
'''
def extrapolate_temp(dataframe):
    ht1 = dataframe['wind_sensor_height_1']
    ht2 = dataframe['wind_sensor_height_2']
    temp_ht1 = dataframe['ta_tc1']
    temp_ht2 = dataframe['ta_tc1']

    surface_temp = temp_ht1 - (((temp_ht2 - temp_ht1)/(ht2 - ht1))*ht1)
    return surface_temp
'''


def gradient_fluxes(df):  # This method is very sensitive to input data quality
    """Returns Sensible Heat Flux and Latent Heat Flux based on Steffen & DeMaria (1996) method"""
    g = 9.81  # m/s**2
    cp = 1005  # J/kg/K
    k = 0.4  # von Karman
    Lv = 2.50e6  # J/kg

    fillvalue = common.fillvalue_float

    ht_low, ht_high, ta_low, ta_high, wspd_low, wspd_high, rh_low, rh_high, phi_m, phi_h = ([] for _ in range(10))

    # Average temp from both sensors for height1 and height2
    ta1 = df.loc[:, ("ta_tc1", "ta_cs1")]
    ta2 = df.loc[:, ("ta_tc2", "ta_cs2")]
    df['ta1'] = ta1.mean(axis=1)
    df['ta2'] = ta2.mean(axis=1)

    # Assign low and high depending on height of sensors
    idx = 0
    while idx < len(df):
        if df['wind_sensor_height_1'][idx] == fillvalue or df['wind_sensor_height_2'][idx] == fillvalue:
            ht_low.append(np.nan)
            ht_high.append(np.nan)
            ta_low.append(df['ta1'][idx])
            ta_high.append(df['ta2'][idx])
            wspd_low.append(df['wspd1'][idx])
            wspd_high.append(df['wspd2'][idx])
            rh_low.append(df['rh1'][idx])
            rh_high.append(df['rh2'][idx])
        elif df['wind_sensor_height_1'][idx] > df['wind_sensor_height_2'][idx]:
            ht_low.append(df['wind_sensor_height_2'][idx])
            ht_high.append(df['wind_sensor_height_1'][idx])
            ta_low.append(df['ta2'][idx])
            ta_high.append(df['ta1'][idx])
            wspd_low.append(df['wspd2'][idx])
            wspd_high.append(df['wspd1'][idx])
            rh_low.append(df['rh2'][idx])
            rh_high.append(df['rh1'][idx])
        else:
            ht_low.append(df['wind_sensor_height_1'][idx])
            ht_high.append(df['wind_sensor_height_2'][idx])
            ta_low.append(df['ta1'][idx])
            ta_high.append(df['ta2'][idx])
            wspd_low.append(df['wspd1'][idx])
            wspd_high.append(df['wspd2'][idx])
            rh_low.append(df['rh1'][idx])
            rh_high.append(df['rh2'][idx])

        idx += 1

    # Convert lists to arrays
    ht_low = np.asarray(ht_low)
    ht_high = np.asarray(ht_high)
    ta_low = np.asarray(ta_low)
    ta_high = np.asarray(ta_high)
    wspd_low = np.asarray(wspd_low)
    wspd_high = np.asarray(wspd_high)
    rh_low = np.asarray(rh_low)
    rh_high = np.asarray(rh_high)
    ps = np.asarray(df['ps'].values)

    # Potential Temperature
    pot_tmp_low = potential_temperature(ps * units.pascal, ta_low * units.kelvin).magnitude
    pot_tmp_high = potential_temperature(ps * units.pascal, ta_high * units.kelvin).magnitude
    pot_tmp_avg = (pot_tmp_low + pot_tmp_high)/2
    ta_avg = (ta_low + ta_high)/2

    # Ri
    du = wspd_high-wspd_low
    du = np.asarray([fillvalue if i == 0 else i for i in du])
    pot_tmp_avg = np.asarray([fillvalue if i == 0 else i for i in pot_tmp_avg])
    ri = g*(pot_tmp_high - pot_tmp_low)*(ht_high - ht_low)/(pot_tmp_avg*du)

    # Phi
    for val in ri:
        if val < -0.03:
            phi = (1-18*val)**-0.25
            phi_m.append(phi)
            phi_h.append(phi/1.3)
        elif -0.03 <= val < 0:
            phi = (1-18*val)**-0.25
            phi_m.append(phi)
            phi_h.append(phi)
        else:
            phi = (1-5.2*val)**-1
            phi_m.append(phi)
            phi_h.append(phi)

    phi_e = phi_h

    # air density
    rho = density(ps * units.pascal, ta_avg * units.kelvin, 0).magnitude  # Use average temperature

    # SH
    ht_low = np.asarray([fillvalue if i == 0 else i for i in ht_low])
    num = np.asarray([-a1 * cp * k**2 * (b1 - c1) * (d1 - e1) for a1, b1, c1, d1, e1 in
           zip(rho, pot_tmp_high, pot_tmp_low, wspd_high, wspd_low)])
    dnm = [a2 * b2 * np.log(c2 / d2)**2 for a2, b2, c2, d2 in
           zip(phi_h, phi_m, ht_high, ht_low)]
    dnm = np.asarray([fillvalue if i == 0 else i for i in dnm])
    sh = num/dnm
    sh = [fillvalue if abs(i) >= 100 else i for i in sh]

    # Specific Humidity
    mixing_ratio_low = mixing_ratio_from_relative_humidity(rh_low, ta_low * units.kelvin, ps * units.pascal)
    mixing_ratio_high = mixing_ratio_from_relative_humidity(rh_high, ta_high * units.kelvin, ps * units.pascal)
    q_low = specific_humidity_from_mixing_ratio(mixing_ratio_low).magnitude
    q_high = specific_humidity_from_mixing_ratio(mixing_ratio_high).magnitude
    q_low = q_low/100  # Divide by 100 to make it in range [0,1]
    q_high = q_high/100

    # LH
    num = np.asarray([-a1 * Lv * k**2 * (b1 - c1) * (d1 - e1) for a1, b1, c1, d1, e1 in
           zip(rho, q_high, q_low, wspd_high, wspd_low)])
    dnm = [a2 * b2 * np.log(c2 / d2)**2 for a2, b2, c2, d2 in
           zip(phi_e, phi_m, ht_high, ht_low)]
    dnm = np.asarray([fillvalue if i == 0 else i for i in dnm])
    lh = num/dnm
    lh = [fillvalue if abs(i) >= 100 else i for i in lh]

    return sh, lh


def nead2nc(args, input_file, output_file, stations):
    """Main function to convert NEAD ascii file to netCDF"""
    df, temperature_vars, pressure_vars = init_dataframe(args, input_file)
    ds = nead.read(input_file)
    station_number = ds.attrs['station_id']

    df['year']=pd.to_datetime(df.timestamp.values).year
    df['hour']=pd.to_datetime(df.timestamp.values).hour
    
    ds = xr.Dataset.from_dataframe(df)
    

    common.log(args, 2, 'Retrieving latitude, longitude and station name')
    latitude, longitude, station_name = get_station(args, input_file, stations)

    print(latitude, longitude, station_name)
    
    common.log(args, 3, 'Calculating time and sza')
    month, day, hour, minutes, time, time_bounds, sza, az, first_date, last_date = get_time_and_sza(
        args, df, longitude, latitude)

    if args.flx:
        common.log(args, 4, 'Calculating Sensible and Latent Heat Fluxes')
        sh, lh = gradient_fluxes(df)
        ds['sh'] = 'time', sh
        ds['lh'] = 'time', lh

    if args.no_drv_tm:
        pass
    else:
        ds['month'] = 'time', month
        ds['day'] = 'time', day
        ds['hour'] = 'time', hour
        ds['minutes'] = 'time', minutes

    ds['time'] = 'time', time
    ds['time_bounds'] = ('time', 'nbnd'), time_bounds
    ds['sza'] = 'time', sza
    ds['az'] = 'time', az
    ds['station_number'] = tuple(), station_number
    ds['station_name'] = tuple(), station_name
    ds['latitude'] = tuple(), latitude
    ds['longitude'] = tuple(), longitude
    # ds['surface_temp'] = 'time', surface_temp

    rigb_vars = []
    if args.rigb:
        ds, rigb_vars = common.call_rigb(
            args, station_name, first_date, last_date, ds, latitude, longitude, rigb_vars)

    comp_level = args.dfl_lvl

    global columns
    columns = df.columns.values
    
    common.load_dataset_attributes('nead', ds, args, rigb_vars=rigb_vars, 
                                   temperature_vars=temperature_vars,
                                   pressure_vars=pressure_vars)
    
    encoding = common.get_encoding('nead', common.get_fillvalue(args), comp_level, args)

    common.write_data(args, ds, output_file, encoding)
