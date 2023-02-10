# -*- coding: utf-8 -*-
"""
L12L2_translation
Script to convert L1 Files in L2.
All files in folder_input will be converted in L2 and saved in folder_output.
The folder containing Lidar constants is described by the variable folder_calibration 
Data format Documentation: E-PROFILE_ALC_data_format_v06-1.docx
Raw2L1 and L12L2 current issues: Raw2L1-L12L2_Issues_and_developments_v1-1.docx
Created on Fri Oct 21 09:48:15 2016
@authors: Maxime Hervo, Rolf Ruefenacht (MeteoSwiss)
"""

from __future__ import print_function, division, absolute_import
from builtins import zip
from builtins import next
from builtins import range
from builtins import object
from netCDF4 import Dataset, num2date
import numpy as np
from scipy.constants import lambda2nu

import contextlib
import datetime
import os
import sys



# this version string is maintained by bumpversion, do not edit manually
VERSION = '2.0.7.dev1'

KNOWN_INSTRUMENTS = {
    'CL31': { 'default_calibration': 1e8, 'calibration_units': 'm^3*sr*V'},
    'CL51': { 'default_calibration': 1e8, 'calibration_units': 'm^3*sr*V' },
    'CHM15k': { 'default_calibration': 3e11, 'calibration_units': 'm^3*sr*counts/s'},
    'Mini-MPL': { 'default_calibration': 5e5, 'calibration_units': '1E6*m^3*sr*MHz/uJ'},
    # FIXME: OIC-743: we will need to add CS135 to this dict when those instruments are ingested. 
}

ERROR_STRFTIME_FORMATTER = '%d/%m/%y %H:%M:%S'
METSWITCH_FILENAME_TIMESTAMP_FORMATTER = '%m%d%H%M'


def check_instrument_type_is_supported(instrument_type):
    if instrument_type not in KNOWN_INSTRUMENTS:
        raise NotImplementedError(" This instrument type is not implemented: " + instrument_type)


def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def get_calibration(instrument_type, calibration_file=None):
    '''
    Returns a calibration, either read from file
    or a default for the instrument type
    '''
    if calibration_file is not None:
        data_cal = Dataset(calibration_file)

        calibration_all = data_cal.variables['lidar_constant']
        calibration = np.nanmedian(calibration_all)
        data_cal.close

        return calibration
    else:
        return KNOWN_INSTRUMENTS[instrument_type]['default_calibration']

def oktas_to_proportion(oktas):
    return oktas / 8.

def proportion_to_oktas(proportion):
    return proportion * 8

def calculate_cloud_amount_mean(cloud_amount, index):
    # Some instruments (e.g. Vaisala CL31 and CL51) report multiple layers...
    if len(cloud_amount.shape) > 1:
        if all(np.isnan(cloud_amount[index, 0])): #0th layer is masked/fill value for instruments not reporting cloud_amount
            cloud_amount_mean = np.nan
        else:
            ca_rel = oktas_to_proportion(np.nanmean(cloud_amount[index, 0], 0))
            for layer in range(1, cloud_amount.shape[1]):
                ca_rel = ca_rel + (1 - ca_rel) * oktas_to_proportion(np.nanmean(cloud_amount[index, layer], 0))
            cloud_amount_mean = np.round(proportion_to_oktas(ca_rel), 0)
    # ... others don't.
    else:
        if all(np.isnan(cloud_amount[index])): 
            cloud_amount_mean = np.nan
        else:
            cloud_amount_mean = np.round(np.mean(cloud_amount[index], 0))

    return cloud_amount_mean

def calculate_vertical_visibility_mean(vertical_visibility, index):
    """Calculate mean vertical visibility (a scalar).
    Given an array of `vertical_visibility` and an `index` into its
    first dimension (time), calculates the mean vertical visibility at
    that time, rounded to the nearest integer.
    The `index` should be a NumPy array of bools, e.g.
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> index = np.array([False, False, False, True, True, True])
    >>> calculate_vertical_visibility_mean(data, index)  # (4 + 5 + 6) / 3
    5.0
    """
    return np.round(np.mean(vertical_visibility[index]))


def populate_L2_netCDF(ncID, input_file, calibration_file=None,
                       resol_time=5, resol_altitude=30, use_temp_file=True):
    """Populate an empty L2 NetCDF dataset `ncID` from L1 NetCDF.
    Given the `input_path` of the L1 NetCDF file, perform processing
    of L1 data into L2 and write it to the open netCDF Dataset object
    `ncID`.
    Args:
      ncID:              empty L2 netCDF Dataset object
      input_file:        L1 netCDF file path
      calibration_file:  single netCDF calibration file to read (or None to use defaults)
      resol_time:        time resolution (minutes) for the regridded/downsampled output 
      resol_altitude:    vertical resoution (metres) for the regridded/downsampled output
      use_temp_file:     write the entire output to a temporary file and then rename to output_file
    """

    print(input_file)

    # %% Read DAta
    data = Dataset(input_file)

    # Read attributes
    instrument_id = data.instrument_id
    instrument_firmware_version = data.instrument_firmware_version
    instrument_type = data.instrument_type

    check_instrument_type_is_supported(instrument_type)

    station_latitude = data.variables['station_latitude'][:]
    station_longitude = data.variables['station_longitude'][:]
    station_altitude = data.variables['station_altitude'][:]

    # Read time
    time = data.variables['time'][:]

    # Read range and altitude
    range_alc = data.variables['range'][:]
    z = range_alc + station_altitude

    # Read Range corrected signal
    rcs = data.variables['rcs_0'][:]

    # Read Cloud base height
    cloud_base_height = data.variables['cloud_base_height'][:]

    cloud_base_height = cloud_base_height.astype(float)
    if not(type(cloud_base_height) == np.ndarray):
        # Convert masked array to Classic arrays
        cloud_base_height = cloud_base_height.filled(np.nan)
    cloud_base_height[cloud_base_height < 0] = np.nan

    # Read Cloud cover
    cloud_amount = data.variables['cloud_amount'][:]
    
    cloud_amount = cloud_amount.astype(float)
    if not(type(cloud_amount) == np.ndarray):
        # Convert masked array to Classic arrays
        # fill up with nans instead of zeros which triggered wrong total cloud amounts before in case of nodata
        cloud_amount = cloud_amount.filled(np.nan) 
    
    tilt_angle = data.variables['tilt_angle'][:]
    wavelength = data.variables['l0_wavelength'][:]

    # Read Vertical Optical Range
    # The variable id reported as 'vor' in CHM15k L1 files and 'vertical_visibility' in Vaisala instruments
    # This information is not found in mini-MPL data.

    if instrument_type == 'CL31':
        vertical_visibility = data.variables['vertical_visibility'][:]
    elif instrument_type == 'CL51':
        vertical_visibility = data.variables['vertical_visibility'][:]
    elif instrument_type == 'CHM15k':
        vertical_visibility = data.variables['vor'][:]
        # if vertical visibility is not available we set the attribute as missing then create an empty array np.nan.
    else:
        vertical_visibility = nans(len(time))

    vertical_visibility = vertical_visibility.astype(float)
    if not(type(vertical_visibility) == np.ndarray):
        # Convert masked array to Classic arrays
        vertical_visibility = vertical_visibility.filled(np.nan)

    # %% Average Data
    
    # if only 1 value in the time array then no need to compute the average.
    # This will occur when the time resolution of the L1 data is the same as resol_time.
    
    resol_time_days = float(resol_time) / 24.0 / 60.0  # Convert minutes in Days
    if (len(time)) > 1:
        time_mean = np.flip(np.arange(max(time), min(time), -resol_time_days))  # keep END time of L1 bunches as time for L2
    else:
        time_mean = time

    range_alc_mean = np.arange(min(range_alc), max(range_alc), resol_altitude)

    rcs_mean_tmp = nans((len(time_mean), len(range_alc)))
    rcs_mean = nans((len(time_mean), len(range_alc_mean)))

    cbh_mean = nans((len(time_mean), 3))
    cloud_amount_mean = nans((len(time_mean), 1))

    vertical_visibility_mean = nans((len(time_mean),))
    
    # Additional time tolerence implemented to resolve issues with MPL instrument - see OIC-1709 
    time_sel_tolerance = 1e-5  # in days. 1e-5 corresponds to a bit more than 1s
    for t in range(0, len(time_mean)):
        index = np.logical_and(time > time_mean[t] - resol_time_days - time_sel_tolerance,
                               time <= time_mean[t] + time_sel_tolerance)

        rcs_mean_tmp[t, :] = np.mean(rcs[index, :], axis=0)

        # try except block as code can fail here
        # believed to be due to corrupted cloud height values in the L1 file.
        try:
            # Consider that the cloud base height is the lowest altitude of
            # each cloud layer
            cbh_mean[t, :] = np.nanmin(cloud_base_height[index, 0:3], 0)
        except Exception:
            # We don't want to raise a proper tivoli alert here because there is no useful action than can be taken.
            print(
                datetime.datetime.now().strftime(ERROR_STRFTIME_FORMATTER) + " LIDARNET ALL ALL HERMES_E-PROFILE DATA_ERROR 121 HARMLESS 1 FALSE  UNABLE TO CALCULATE CLOUD BASE HEIGHT MEAN " + input_file)
            # We swallow the exception and continue processing this file.

        cloud_amount_mean[t] = calculate_cloud_amount_mean(cloud_amount, index)

        vertical_visibility_mean[t] = calculate_vertical_visibility_mean(vertical_visibility, index)

    for i in range(0, len(range_alc_mean)):
        index = np.logical_and(range_alc >= range_alc_mean[i],
                               range_alc < range_alc_mean[i] + resol_altitude)
        if not any(index):  # avoid gaps in grid when range_alc is coarser than range_alc_mean (typically for MPL)
            index = [np.abs(range_alc - range_alc_mean[i]).argmin()]  # index as list to preserve dimension when slicing
            pass
        rcs_mean[:, i] = np.mean(rcs_mean_tmp[:, index],  axis=1)

    print("Get Calibration info:")
    print(instrument_type, calibration_file)
    
    calibration = get_calibration(instrument_type, calibration_file)
    
    print(calibration)

    # Apply calibration and convert to mm^-1
    attenuated_backscatter = rcs_mean / calibration * 1e6

    # %% Calculate flags and uncertainties
    attenuated_backscatter_uncertainties = abs(
        attenuated_backscatter * 0.25)
    cbh_uncertainties = np.empty((len(time_mean), 3),  dtype=float)
    cbh_uncertainties.fill(50)
    flag = np.zeros(np.shape(attenuated_backscatter))
    is_cloud = ~np.isnan(cbh_mean)
    if is_cloud.any():
        for i in range(0, len(time_mean)):
            flag[i, range_alc_mean > max(cbh_mean[i] + 1000)] = 1

    # Global attributes from L1 NetCDF
    nc_attrs = {attr: data.getncattr(attr) for attr in data.ncattrs()}

    data.close()

    # dimensions.
    nc_time = ncID.createDimension('time', None)
    nc_altitude = ncID.createDimension('altitude', len(range_alc_mean))
    nc_layer = ncID.createDimension('layer', 3)

    # Copy Global attibute
    for nc_attr in nc_attrs:
        ncID.setncattr(nc_attr, nc_attrs[nc_attr])
    ncID.history = ncID.history + " / " + datetime.datetime.now().strftime("%Y%m%d") + ' L12L2 ' + VERSION

    # Create variables.
    # You must use zlib=true to preserve an significant amount of disk
    # space
    nc_time = ncID.createVariable('time', 'f8', ('time',), zlib=True)
    nc_time.long_name = "End time (UTC) of the measurements"
    nc_time.units = "days since 1970-01-01 00:00:00.000"
    nc_time.long_name = "Time (UTC) of the calibration period"
    nc_time.standard_name = "time"
    nc_time.calendar = "gregorian"
    nc_time[:] = time_mean

    nc_start_time = ncID.createVariable(
        'start_time', 'f8', ('time',), zlib=True)
    nc_start_time.long_name = "Start time (UTC) of the measurements"
    nc_start_time.units = "days since 1970-01-01 00:00:00.000"
    nc_start_time[:] = time_mean - resol_time_days

    nc_altitude = ncID.createVariable(
        'altitude', 'f8', ('altitude',), zlib=True)
    nc_altitude.long_name = "Altitude above sea level"
    nc_altitude.units = "m"
    nc_altitude.standard_name = "altitude"
    altitude = np.mean(
        np.cos(np.deg2rad(tilt_angle))) * range_alc_mean + station_altitude
    nc_altitude[:] = altitude

    nc_latitude = ncID.createVariable(
        'latitude', 'f8', ('altitude', 'time'), zlib=True)
    nc_latitude.long_name = "Latitude for each measurement"
    nc_latitude.units = "degree_north"
    fake_lat = np.empty(
        (len(range_alc_mean), len(time_mean)),  dtype=float)
    fake_lat.fill(np.nan)
    nc_latitude[:, :] = fake_lat

    nc_longitude = ncID.createVariable(
        'longitude', 'f8', ('altitude', 'time'), zlib=True)
    nc_longitude.long_name = "Longitude for each measurement"
    nc_longitude.units = "degree_east"
    nc_longitude[:, :] = fake_lat

    nc_beta_att = ncID.createVariable(
        'attenuated_backscatter_0', 'f8', ('altitude', 'time'), zlib=True)
    nc_beta_att.long_name = "Attenuated Backscatter at wavelength 0"
    nc_beta_att.units = "1E-6*1/(m*sr)"
    nc_beta_att[:, :] = np.transpose(attenuated_backscatter)

    nc_un_beta_att = ncID.createVariable(
        'uncertainties_att_backscatter_0', 'f8', ('altitude', 'time'), zlib=True)
    nc_un_beta_att.long_name = "Uncertainties for Attenuated Backscatter at wavelength 0"
    nc_un_beta_att.units = "1E-6*1/(m*sr)"
    nc_un_beta_att[:, :] = np.transpose(
        attenuated_backscatter_uncertainties)

    nc_wavelength = ncID.createVariable('l0_wavelength', 'f8', zlib=True)
    nc_wavelength.long_name = "Wavelength of Laser for channel 0"
    nc_wavelength.units = "nm"
    nc_wavelength[:] = wavelength

    nc_station_longitude = ncID.createVariable(
        'station_longitude', 'f8', zlib=True)
    nc_station_longitude.long_name = "Longitude of measurement station"
    nc_station_longitude.units = "degrees_east"
    nc_station_longitude.standard_name = "longitude"
    nc_station_longitude[:] = station_longitude

    nc_station_latitude = ncID.createVariable(
        'station_latitude', 'f8', zlib=True)
    nc_station_latitude.long_name = "Latitude of measurement station"
    nc_station_latitude.standard_name = "latitude"
    nc_station_latitude.units = "degrees_north"
    nc_station_latitude[:] = station_latitude

    nc_station_altitude = ncID.createVariable(
        'station_altitude', 'f8', zlib=True)
    nc_station_altitude.long_name = "Altitude of measurement station"
    nc_station_altitude.units = "m"
    nc_station_altitude[:] = station_altitude

    nc_flag = ncID.createVariable(
        'quality_flag', 'i8', ('altitude', 'time'), zlib=True)
    nc_flag.long_name = "Attenuated Backscatter Quality flag"
    nc_flag.comments = "flag_values: 0,1,2.  flag_meanings: 0: valid data;  1: do_not_use; 2: no_information"
    nc_flag.flag_values = np.array((0, 1, 2))
    nc_flag.flag_meanings = "valid invalid unknown"
    nc_flag[:] = np.transpose(flag)

    nc_vertical_visibility = ncID.createVariable(
        'vertical_visibility', 'f8', ('time',), zlib=True)
    nc_vertical_visibility.long_name = "Vertical Visibility"
    nc_vertical_visibility.units = "m"
    nc_vertical_visibility[:] = vertical_visibility_mean

    nc_cloud_base_height = ncID.createVariable(
        'cloud_base_height', 'f8', ('time', 'layer'), zlib=True)
    nc_cloud_base_height.long_name = "Cloud Base Height above ground level"
    nc_cloud_base_height.units = "m"
    nc_cloud_base_height[:] = cbh_mean

    nc_cbh_uncertainties = ncID.createVariable(
        'cbh_uncertainties', 'f8', ('time', 'layer'), zlib=True)
    nc_cbh_uncertainties.long_name = "Uncertainties for Cloud Base Height"
    nc_cbh_uncertainties.units = "m"
    nc_cbh_uncertainties[:] = cbh_uncertainties

    nc_cloud_amount = ncID.createVariable(
        'cloud_amount', 'f8', ('time',), zlib=True)
    nc_cloud_amount.long_name = "Cloud amount in octa"
    nc_cloud_amount.units = "1"
    nc_cloud_amount[:] = cloud_amount_mean

    nc_calibration_constant_0 = ncID.createVariable(
        'calibration_constant_0', 'f8', ('time',), zlib=True)
    nc_calibration_constant_0.long_name = "Value of the calibration constant used to calculate the attenuated back scatter at wavelength 0"

    nc_calibration_constant_0.units = KNOWN_INSTRUMENTS[instrument_type]['calibration_units']

    nc_calibration_constant_0[:] = calibration


def get_output_gts_bufr_filename(ttaaii, instrument_id, message_time, dirname, basename):
    if not dirname:
        print("Not generating GTS BUFR: --output_gts_bufr_dir not specified.")
        return

    if not basename:
        if not ttaaii:
            print("Not generating GTS BUFR: TTAAii not specified in headers.")
            return

        if not instrument_id:
            print("Not generating GTS BUFR: instrument ID not set.")
            return

        timestamp = message_time.strftime(METSWITCH_FILENAME_TIMESTAMP_FORMATTER)
        basename = '{}_EUOP_{}_{}.PRO'.format(ttaaii, instrument_id, timestamp)

    return os.path.join(dirname, basename)


def write_L2_BUFR(path, data, wmo_id, storm_events):
    with atomic_file_output(path) as tmp:
        ALCBUFREncoder(wmo_id, logProductEvent=storm_events).write(data, tmp.name)


def hermes_processing():
    """ Hermes processing entry point 
    
    Example usage when installed in a virtualenv (see also setup.py):
        eprofile_alc_1_to_2 -i <inputfile> -o <outputfile>
    """
    import argparse
    parser = argparse.ArgumentParser(description='Eprofile ALC L1 to L2 translation (Hermes Integration, version %s)' % VERSION)
    parser.add_argument('-i','--input_file', help='input L1 eprofile alc netCDF file', required=True)
    parser.add_argument('-o','--output_file', help='output L2 eprofile alc netCDF file', required=True)
    parser.add_argument('-c','--calibration_file', help='input netcdf calibration file', default='none')
    parser.add_argument('-T','--time_resolution', type=float, help='output time resolution (minutes)', default=5.)
    parser.add_argument('-A','--alt_resolution', type=float, help='output altitude resolution (metres)', default=30.) 
    parser.add_argument('--no_storm_events',  dest='no_storm_events', action='store_true', help='do NOT emit Storm events. Default: emit Storm events')
    parser.add_argument('--json_headers', dest='headers', help='message headers', metavar='JSON', type=Headers, default=Headers())
    parser.add_argument('--output_gts_bufr_dir', default='',
                        help='Directory to write GTS BUFR to (if not specified, GTS BUFR will not be written)')
    parser.add_argument('--output_gts_bufr_filename', default='',
                        help='Filename (basename) to write GTS BUFR to (if not specified, '
                             'an automatic filename is inferred from the data)')

    args = parser.parse_args()

    # Because it is too difficult/impossible to make the form of arguments passed to this command by the Camel context conditional.
    if os.path.basename(args.calibration_file) == 'none':
       cal_file = None
    else:
       cal_file = args.calibration_file


    storm_event_nc = storm_product_event(productType=PRODUCT_TYPE_NETCDF,
                                         filePath=args.output_file,
                                         headers=args.headers,
                                         autoLog=not args.no_storm_events)

    with storm_event_nc:
        # The Storm event relates to the final output file, so we
        # can't exit this context until the temporary file is
        # renamed. And this can't happen until we've finished using
        # it to create the BUFR. Therefore the NetCDF product event
        # interval will have to include the entire BUFR product
        # event interval; as long as we use the L2 NetCDF as the
        # source for the L2 BUFR we cannot have the Storm events
        # happen one after the other.

        with atomic_file_output(args.output_file) as tmp:
            # Wrapping everything in the atomic_file_output context
            # because we don't want our output NetCDF file to disappear
            # while we're still using it to generate the BUFR output.
            # Instead, we will generate the BUFR from the NetCDF in its
            # temporary location and move it only after we have finished
            # with it.

            with contextlib.closing(Dataset(tmp.name, 'w')) as nc_level2:
                populate_L2_netCDF(nc_level2, args.input_file, cal_file,
                                   args.time_resolution, args.alt_resolution)

                validity_time = num2date(nc_level2.variables['time'][-1],
                                         nc_level2.variables['time'].units)

                bufr_path = get_output_gts_bufr_filename(
                    args.headers.instrumentGtsOutgoingTTAAII,
                    args.headers.instrumentAssetId,
                    validity_time,
                    args.output_gts_bufr_dir,
                    args.output_gts_bufr_filename)

                # There is no Storm event or atomic file output
                # stuff here: the BUFR encoder handles that for us.
                if bufr_path is not None:
                    if not args.headers.networkBlocking['hermes_eprofile_alc']:
                        write_L2_BUFR(
                            bufr_path,
                            nc_level2,
                            args.headers.siteWmoId,
                            not args.no_storm_events
                        )
                    else:
                        print('Not generating gts BUFR; data blocking flag is set')

            # Set file actuals on the temporary file instead of the final file
            # to avoid a race with, e.g, DART.
            storm_event_nc.update_file_actuals(tmp.name)
