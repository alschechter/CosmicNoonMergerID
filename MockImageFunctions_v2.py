import astropy.io.fits as pyfits
import numpy as np
from sedpy import observate
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.constants as const
from scipy import ndimage
from astropy.convolution import convolve
from astropy import wcs
import astropy.units as u
import pandas as pd
from regions.core import PixCoord
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.wcs as w
from glob import glob
from astropy.nddata.utils import Cutout2D
import os.path 
import sys
import matplotlib
matplotlib.use('agg')


#read in image and wavelengths
tngimage = pyfits.open('dusty_img5_total.fits') #z=1.5


tngwav = tngimage[1].data

tngim = tngimage[0].data
print('original skirt', np.shape(tngim))

#print(wavelengths)
tng50_arcsecperpixel = tngimage[0].header['CDELT1'] * u.arcsec 
print(tng50_arcsecperpixel)

tng50_radperpixel = tng50_arcsecperpixel.to(u.rad)
tng50srperpixel = (tng50_radperpixel**2).to(u.sr)
#print(tng50_arcsecperpixel)
z10Mpc = z_at_value(cosmo.luminosity_distance, 10*1e3 * u.kpc)
#print(z10Mpc)

ACS = [ 'acs_wfc_f435w', 'acs_wfc_f475w' , 'acs_wfc_f555w' , 'acs_wfc_f606w', 'acs_wfc_f625w' , 'acs_wfc_f775w' , 'acs_wfc_f814w']
WFC3_IR = ['wfc3_ir_f105w' , 'wfc3_ir_f110w' , 'wfc3_ir_f125w' , 'wfc3_ir_f140w' , 'wfc3_ir_f160w']
WFC3_UVIS = ['wfc3_uvis_f275w' , 
        'wfc3_uvis_f336w' , 'wfc3_uvis_f475w' , 'wfc3_uvis_f555w' , 'wfc3_uvis_f606w' , 'wfc3_uvis_f814w']

CANDELSfilters = ['acs_wfc_f435w', 'acs_wfc_f606w', 'acs_wfc_f775w' , 'acs_wfc_f814w', 'acs_wfc_f850lp', 
                  'wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f160w']

LSSTfilters = ['lsst_baseline_i', 'lsst_baseline_r']

Romanfilter = ['roman_wfi_f158']

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def angle_per_pixel_at_redshift(z):
    kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(z)
    kpc_per_arcsec = kpc_per_arcmin.to(u.kpc/u.arcsec)
    
    kpc_per_arcmin_10Mpc = cosmo.kpc_proper_per_arcmin(z10Mpc)
    kpc_per_arcsec_10Mpc = kpc_per_arcmin_10Mpc.to(u.kpc/u.arcsec)
    kpc_per_pixel_10Mpc = kpc_per_arcsec_10Mpc * tng50_arcsecperpixel

    arcsec_per_pixel_atz = kpc_per_pixel_10Mpc / kpc_per_arcsec
    print('arcsec at z', arcsec_per_pixel_atz)
    #print(arcsec_per_pixel_atz*500)
    sr_per_pixel_atz = (arcsec_per_pixel_atz.to(u.radian)**2).to(u.sr) 
    
    return arcsec_per_pixel_atz, sr_per_pixel_atz

#print(arcsec_per_pixel_at_redshift(1))

def GetBroadband(ImageFileName):

    tngimage = pyfits.open(ImageFileName)
 
    tngwav = tngimage[1].data
    tngim = tngimage[0].data
    #print('tngim: ', np.shape(tngim))
    tngim_sum = np.sum(tngim, axis = 0)
    #for pixel size checking
    print('Shape of SKIRT image', np.shape(tngim_sum))
    
    return tngwav, tngim, tngim_sum 



def PlaceAtRedshift(ImageFileName, z):

    tngwav, tngim, tngim_sum = GetBroadband(ImageFileName)
    
    #un PLACE AT 10Mpc AWAY
    I_z0 = tngim * (1+z10Mpc)**3 # image is in MJy/sr so we un-redshift it
    #print('I_z0: ', np.shape(I_z0))
    tngwav = tngwav.field(0) * (1. + z10Mpc)
    
    #GO TO DESIRED z
    I_dimmed = I_z0 * (1+z)**(-3) ##-3 BECAUSE WE'RE IN FREQUENCY SPACE?
    #print('I_z0: ', np.shape(I_z0))
    tngwav_redshifted = (tngwav * u.micron * (1+z)).to(u.AA)
    #tngwav_redshifted_frequency = const.c.to(u.AA/u.s) / tngwav_redshifted
    tngwav_redshifted_frequency = (const.c.to(u.AA/u.s) / tngwav_redshifted).to(u.Hz)
    
    return tngwav_redshifted, I_dimmed, tngwav_redshifted_frequency 

def FilterEntireImage(DimmedImage, redshifted_wavelengths, redshifted_wavelengths_frequency, hstfilter, z):
    filters = observate.load_filters([hstfilter])
    dimensions = DimmedImage.shape # [nspec, ncol, nrow]
    #print('DimmedImage: ', np.shape(DimmedImage))
    DimmedImage_unraveled = DimmedImage.reshape([dimensions[0], -1]).T  # shape = (ncol*nrow, nspec)
    DimmedImage_unraveled = DimmedImage_unraveled * u.MJy / u.sr #this is a surface brightness
    # now we need a flux density
    Dimmed_Image_unraveled_nosr = DimmedImage_unraveled * angle_per_pixel_at_redshift(z)[1] #units of MJy
    #redshifted_wavelengths_cm = redshifted_wavelengths.to(u.cm)
    #now to go erg/s/cm^2/AA for sedby getSED
    FluxDensity = Dimmed_Image_unraveled_nosr.to(u.erg/u.s/u.cm**2/u.AA, equivalencies = u.spectral_density(redshifted_wavelengths))

    mags = observate.getSED(redshifted_wavelengths.value, FluxDensity.value, filterlist = filters) # AB mags has shape (ncol * nrow, nfilt) #assumes AA
    mags = mags * u.ABmag
    finite = mags[np.isfinite(mags)]
    finite_top = finite[finite < np.percentile(finite, 5)]
    average_of_finite = np.average(finite_top)
    print('ABmag', average_of_finite)
    Brightness_Jy = mags.to(u.Jy).T 
    Brightness_Jy_sr = Brightness_Jy / angle_per_pixel_at_redshift(z)[1] #need surface brightness for rebinning below
    Brightness_Jy_sr_image = Brightness_Jy_sr.reshape([len(filters), dimensions[1], dimensions[2]])  # this has shape (nfilt, ncol, nrow)
    print('Dimmed Image', np.shape(Brightness_Jy_sr_image))
    return Brightness_Jy_sr_image.value

    #now can remove angular dependence and convert to erg/s/cm/AA **** angular dependence now CAMERA pixel angle because we rebinned
def FluxAndRebin(FilteredImage, redshifted_wavelengths, hstfilter, z):
    if hstfilter in ACS or hstfilter in WFC3_UVIS:
        camera_pixel_scale = 0.03 * u.arcsec #from CANDELS readme
    if hstfilter in WFC3_IR:
        camera_pixel_scale = 0.06 * u.arcsec
    if hstfilter in LSSTfilters:
        camera_pixel_scale = 0.2 * u.arcsec
    if hstfilter in Romanfilter:
        camera_pixel_scale = 0.11 * u.arcsec
    camera_pixel_scale_sr = (camera_pixel_scale.to(u.rad)**2).to(u.sr)
    print('filtered image shape: ', np.shape(FilteredImage[0]))
    #simulation arcsec_per_pixel = 
    simulation_pixel_scale_arcsec, simulation_pixel_scale_sr = angle_per_pixel_at_redshift(z)
    print('sim pixel scale arcsec' ,simulation_pixel_scale_arcsec)    
    rebin_scale = (simulation_pixel_scale_arcsec/camera_pixel_scale).value    
    print('scale: ', rebin_scale)
    tngim_rebin = ndimage.zoom(FilteredImage[0], rebin_scale) * u.Jy/u.sr
    print('Shape Rebinned Image', np.shape(tngim_rebin))
    
    #now can remove angular dependence and convert to erg/s/cm/AA **** angular dependence now CAMERA pixel angle because we rebinned
    tngim_rebin = (tngim_rebin * camera_pixel_scale_sr) 
    #tngim_rebin = (tngim_rebin * 1e-23 * const.c.to(u.AA/u.s).value / redshifted_wavelengths.value )
    
    if hstfilter == 'wfc3_ir_f125w':
        centralwave = 12500
    elif hstfilter == 'wfc3_ir_f160w':
        centralwave = 16000
    elif hstfilter == 'acs_wfc_f606w':
        centralwave = 6060
    elif hstfilter == 'acs_wfc_f814w':
        centralwave = 8140
    elif hstfilter == 'lsst_baseline_i':
        centralwave = 7500
    elif hstfilter == 'lsst_baseline_r':
        centralwave = 6223  
    elif hstfilter == 'roman_wfi_f158':
        centralwave = 15716
    tngim_rebin = (tngim_rebin.to(u.erg/u.s/u.cm**2/u.AA, equivalencies = u.spectral_density(centralwave*u.AA))).value
    #print('trouble shooting units here', tngim_rebin)
    return tngim_rebin

def ConvolvePSF(RebinnedImage, hstfilter):
    psffile = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/PSFs/' + hstfilter + '.fits'
    #print(psffile)
    psf = pyfits.open(psffile)
    psfdata = psf[0].data
    psfdata = psfdata/np.max(psfdata)
    #print(np.shape(psfdata))
    #print(np.shape(psfdata[0:-1,0:-1]))
    if len(psfdata) % 2 == 0:
        ConvImage = convolve(RebinnedImage, psfdata[0:-1,0:-1])
    else: 
        ConvImage = convolve(RebinnedImage, psfdata)
    #print(np.shape(ConvImage))
    
    return ConvImage


def CombineSteps(Subhalo, viewpoint, hstfilter, z, example = False): #must feed filters in a list=
    if os.path.exists('/n/holylfs05/LABS/hernquist_lab/Lab/xshen/galaxy_merger_images/TNG50-1/mergers/output40_' + str(Subhalo) + '/dusty_img' + str(viewpoint) + '_total.fits') == True:
        ImageFileName = '/n/holylfs05/LABS/hernquist_lab/Lab/xshen/galaxy_merger_images/TNG50-1/mergers/output40_' + str(Subhalo) + '/dusty_img' + str(viewpoint) + '_total.fits'
        print('its a merger')
        
        RedshiftedWavelengths, DimmedBroadband, RedshiftedWavelengths_Frequency = PlaceAtRedshift(ImageFileName, z)
        FilteredImage = FilterEntireImage(DimmedBroadband, RedshiftedWavelengths, RedshiftedWavelengths_Frequency, hstfilter, z)

        #np.save('filterex.npy', FilteredImage)
        #np.save('filtered_example' + str(Subhalo) + '_' + str(viewpoint) + str(hstfilter) +'.npy', FilteredImage)
        Rebinned = FluxAndRebin(FilteredImage, RedshiftedWavelengths, hstfilter, z)
        
        Final = ConvolvePSF(Rebinned, hstfilter)
        
    
        #print(Final)
        if example == True:
            np.save('filterex.npy', FilteredImage)
            np.save('rebinex.npy', Rebinned)
            np.save('convolveex.npy', Final)
            
        np.save('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground/' + str(Subhalo) + '_' + str(viewpoint) + str(hstfilter) +'.npy', Final)
        return Final
    
    elif os.path.exists('/n/holylfs05/LABS/hernquist_lab/Lab/xshen/galaxy_merger_images/TNG50-1/nonmergers/output40_' + str(Subhalo) + '/dusty_img' + str(viewpoint) + '_total.fits') == True:
        ImageFileName = '/n/holylfs05/LABS/hernquist_lab/Lab/xshen/galaxy_merger_images/TNG50-1/nonmergers/output40_' + str(Subhalo) + '/dusty_img' + str(viewpoint) + '_total.fits'
        print('its a nonmerger')
        
        RedshiftedWavelengths, DimmedBroadband, RedshiftedWavelengths_Frequency = PlaceAtRedshift(ImageFileName, z)
        FilteredImage = FilterEntireImage(DimmedBroadband, RedshiftedWavelengths, RedshiftedWavelengths_Frequency, hstfilter, z)
        Rebinned = FluxAndRebin(FilteredImage, RedshiftedWavelengths, hstfilter, z)
        Final = ConvolvePSF(Rebinned, hstfilter)
        np.save('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mocks_nobackground/' + str(Subhalo) + '_' + str(viewpoint) + str(hstfilter) + '.npy', Final)
        return Final 
    
    else:
        print('booo')

    
    
def MakeBadList(MosaicFileName, SourceList):
    mosaic_path = MosaicFileName
    mosaic = pyfits.open(mosaic_path)
    #mosaicdata = mosaic[0].data
    #print(np.shape(mosaicdata))
    mosaicheader = mosaic[0].header
    wcsheader = w.WCS(mosaicheader)
    wcsheader.sip = None    
    
    coordtable = pd.read_csv(SourceList)
    RA = coordtable['RA'].values
    Dec = coordtable['Dec'].values
    skycoords = SkyCoord(RA, Dec, unit='deg', frame='icrs')
    pixcoords = PixCoord.from_sky(skycoord=skycoords, wcs=wcsheader)    
    sourcecoords = np.array(list(zip(RA, Dec)))
    sourcecoords = tuple(map(tuple, sourcecoords))
    sourcepixels = np.round(np.array(wcsheader.world_to_pixel_values(sourcecoords)))
    sourcepixels = list(tuple(map(tuple, sourcepixels)))
    #print('sourcepixel: ', sourcepixels[0])
    #print('sourcecoords: ', sourcecoords[0])
    
    badlist_all = []
    for point in sourcepixels:
        #print('point: ', point)
        badlist_singlesource = []
        #x and y make a box --- all points in box are "bad"
        xmin = point[1] - 20
        xmax = point[1] + 21 #so last one is included in arange
        ymin = point[0] - 20
        ymax = point[0] + 21
        
        xarray = np.arange(xmin, xmax, 1)
        yarray = np.arange(ymin, ymax, 1)
        for x in xarray:
            for y in yarray:
                badpoint = (x,y)
                badlist_singlesource.append(badpoint)
        #print('Length of badlist_singlesource: ', len(badlist_singlesource))
        badlist_all.append(badlist_singlesource)
    print('badlist length: ', np.shape(badlist_all))
    
    #pick point for center
    #dec is vertical
    #ra is horizontal
    badlist_all = np.reshape(badlist_all,(np.shape(badlist_all)[0]*len(badlist_singlesource),2))
    np.savetxt('badlist50.txt', badlist_all)
    return badlist_all

#eventually put this in a while loop to make sure we have enough cutouts              
def PerformCutout(MosaicFileName, badlist, hstfilter, z):
    mosaic_path = MosaicFileName
    mosaic = pyfits.open(mosaic_path)
    mosaicdata = mosaic[0].data
    #print(np.shape(mosaicdata))
    mosaicheader = mosaic[0].header
    wcsheader = w.WCS(mosaicheader)
    wcsheader.sip = None    
    
    kpc_per_side = 100*((1+z)**(-1))* u.kpc
    print('kpc_per_side: ', kpc_per_side)
    if hstfilter in ACS or hstfilter in WFC3_UVIS:
        camera_pixel_scale = 0.03 * u.arcsec #from CANDELS readme
    if hstfilter in WFC3_IR:
        camera_pixel_scale = 0.06 * u.arcsec
        #0.03600000001142689 * u.arcsec #arcsec per pix
    print('camera_pixel_scale: ', camera_pixel_scale)
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)
    print('kpc_per_arcsec: ', kpc_per_arcsec)
    arcsec_per_side = kpc_per_side / kpc_per_arcsec
    print('arcsec_per_side:, ', arcsec_per_side)
    #print(arcsec_per_side)
    pixels_per_side = arcsec_per_side / camera_pixel_scale
    print('Pixels per side', pixels_per_side)
    goodlist = []
    goodlist_coords = []
    goodlist_coords_use = []
    RADec_list = []
    while len(glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/wfc3_ir_f125w*.fits')) < 2: #change this to how many we need
        random_RA = np.random.uniform(0,np.shape(mosaicdata)[1])
        random_dec = np.random.uniform(0,np.shape(mosaicdata)[0])
        random_center = [np.round(random_RA), np.round(random_dec)]
        print(np.shape(random_center))
        random_center_coords = np.array(wcsheader.pixel_to_world_values(random_center[0], random_center[1]))
        print('random center: ', random_center)
        print('center coords: ', random_center_coords)
        if random_center in badlist:
            print('source overlap!')
        elif random_center in goodlist:
            print('source overlap!')
        else:
            print('center okay for use!')
            goodlist.append(random_center)
            goodlist_coords.append(random_center_coords)
            cutout = Cutout2D(data = mosaicdata, position = random_center, size = pixels_per_side.value, 
                  wcs = wcsheader, copy = True)
            if cutout.data.all() != 0.:
                photflam = mosaic[0].header['PHOTFLAM']
                hdu = pyfits.PrimaryHDU(cutout.data*photflam)
                hdulist = pyfits.HDUList([hdu])
                hdulist.writeto('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/' + hstfilter + str(truncate(random_center_coords[0], 10)) + '_' + str(truncate(random_center_coords[1], 10)) + '.fits', overwrite = True)
                RADec_list.append(str(truncate(random_center_coords[0], 10)) + '_' + str(truncate(random_center_coords[1], 10)))
                goodlist_coords_use.append(random_center_coords)
    print('goodlist length', len(goodlist_coords_use))
    np.savetxt('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/' + str(hstfilter) + '_RADec50.txt', RADec_list, fmt="%s")
    np.savetxt('goodlist50.txt', goodlist_coords_use)
    return goodlist_coords_use

def PerformMatchingCutouts(MosaicFileName, coordinatelist, hstfilter, z):
    mosaic_path = MosaicFileName
    mosaic = pyfits.open(mosaic_path)
    mosaicdata = mosaic[0].data
    #print(np.shape(mosaicdata))
    mosaicheader = mosaic[0].header
    wcsheader = w.WCS(mosaicheader)
    wcsheader.sip = None    
    RADec_list2 = []
    kpc_per_side = 100*((1+z)**(-1))* u.kpc
    if hstfilter in ACS or hstfilter in WFC3_UVIS:
        camera_pixel_scale = 0.03 * u.arcsec #from CANDELS readme
    if hstfilter in WFC3_IR:
        camera_pixel_scale = 0.06 * u.arcsec
        #0.03600000001142689 * u.arcsec #arcsec per pix
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)
    arcsec_per_side = kpc_per_side / kpc_per_arcsec
    #print(arcsec_per_side)
    pixels_per_side = arcsec_per_side / camera_pixel_scale
    failed_center = []
    for c in range(len(coordinatelist)):
        coordpair = SkyCoord(coordinatelist[c][0], coordinatelist[c][1], unit="deg")
        print(coordpair)
        cutout = Cutout2D(data = mosaicdata, position = coordpair, size = pixels_per_side.value, 
                  wcs = wcsheader, copy = True)
        if cutout.data.all() != 0.:
            print('cutout created')
            photflam = mosaic[0].header['PHOTFLAM']
            hdu = pyfits.PrimaryHDU(cutout.data*photflam)
            hdulist = pyfits.HDUList([hdu])
            hdulist.writeto('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/' + hstfilter + str(np.round(coordpair.ra.value, decimals = 10)) + '_' + str(np.round(coordpair.dec.value, decimals = 10)) + '.fits', overwrite = True)
            RADec_list2.append(str(truncate(coordpair.ra.value, 10)) + '_' + str(truncate(coordpair.dec.value, 10)))
            print(str(truncate(coordpair.ra.value, 10)) + '_' + str(truncate(coordpair.dec.value, 10)))

        else:
            print('this point does not work')
            failed_center.append(coordpair)
            #write line ti
    np.savetxt('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/' + str(hstfilter) + '_RADec50.txt', RADec_list2, fmt="%s")
    return failed_center
