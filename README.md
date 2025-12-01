# AA272_Final_Project
Florian and Will's AA272 Final Project: Building Height Prediction from Multipath Interference

### Links
**Link to Powerpoint: https://office365stanford-my.sharepoint.com/:p:/g/personal/wcovingt_stanford_edu/EYb38ncef4pEqprw5EN6MIcBYG9xnKiYtl5pA-E7kpCO0w?e=BFNMCm**

### Code Rundown
Lots of scripts and code to go through, so here are the highlights:

**main.py**: this is just the main script. Currently it's meant to coordinate and visualize code from the GNSS Logger app, but go ahead and add on whatever else you need. 

**data_organization.py**: this script does exactly as its name implies. The gnss_lib_py library is good for taking information from the GNSS Logger app, but it does a poor job of coordinating and putting it all together. This does just that. Two of its main return arguments are ["txt_full"] (just a reformatted version of the txt file being fed in) and ["satellite_dataframes"] (shortened to sat_dfs in main). sat_dfs is a set of panda dataframes which includes the data for each satellite being read in, and txt_full is honestly kinda useless at this point I think.

**visualize_interaction.py**: this script takes in the sat_dfs object from data_organization.py and shows it all off in one interactable window. This is meant to help you see the data you're actually working with -- like CN0, PR, PRR, etc. -- before drawing any conclusions. 
One part that I think is really helpful is that sometimes you get data that's just all over the place. What you can do is find the satellite which corresponds to that data on the skyplot by doing some visual color-matching and then switching it off using the tabs on the left.

Those are the main files so far. Here are some extras that aren't as important but that I've kind of just been messing around with. 

**nominal_orbit_propogation.py**: this is a numerical solver of a bunch of GNSS satellites paths given some generic orbital parameters for GNSS-satellite constellations and your current locations (lat, long, alt). I'm planning on using this to try and calculate the mean time for a satellite to cross based on distance from some obstruction, so it's not entirely accurate but I think it's accurate enough for the time being.

**building_height_estimator.py**: this script filters the dataframes for all satellites based on a       maximum elevation angle and C/N0 (determined heuristically), and uses this noisy data to estimate the heights of any surrounding buildings. This is done by defining the origin of a local East-North-Up (ENU) frame at the receiver's location in the urban canyon. The canyon "floor" is defined as a 2D polygon that surrounds the receiver. The algorithm then uses the azimuths in the filtered satellite data to define the heading angle from the receiver to the satellite and calculates the horizontal distance to the first obstruction in that direction. The height of the obstruction can then be estimated using the tangent of the satellite's elevation angle.

**compute_distance_to_walls.py**: this file contains many of the helper functions / utilities used by building_height_estimator.py, including coordinate conversions from lla to ENU, and functions to determine the geometric intersection between the receiver's heading unit vector and the perimiter of the urban canyon polygon.


### File Organization
The only reason there's a file organization section is because it's important for main.py. When you get a set of files from the GNSS Logger app, those files will have the naming scheme gnss_log_YEAR_MONTH_DAY_HOUR_MINUTE_SECOND.txt/nmea/25o. Those files should be placed in a file whose name is just the YEAR_MONTH_DAY_... part -- this way when it's pulled into main, you can edit the directory variable to fit whatever dataset you want to look at and everything should still work.

Also, the rinex_nav_files folder contains .25n files. GNSS Logger Pro outputs .25o files, which correspond to RINEX Observation files -- what your phone sees. RINEX Navigation files instead correspond to some of the ephemeris data which satellites send down. Those files (particularly .25n files) can be found online here: https://cddis.nasa.gov/archive/gnss/data/daily/ (you probably need to make an account). These files _might_ be used for something else later, I don't really know.

### How GNSS Logger Pro Works
There's a bit of a write up on this in the powerpoint, but here's the main idea. First, make sure your phone's connected to the internet (probably to like eduroam if you're on campus -- your home wifi is fine if not). You can make sure you are and the phone's working by checking the 'Plots' and 'Skyplot' tabs; you should see them showcasing live updates. Once that's working, head to the 'Home' tab on the far left and switch as many of the features to 'On' as you can. It's specifically important that you have GNSS Location, Measurements, Network Location, Navigation Messages, GNSS Status, Log NMEA, and Log RINEX on. Once those are flipped to on, go ahead and swipe to the 'Log' tab and hit 'Start Log'. Your phone should vibrate twice and that means it's doing its thing. I recommend starting a log, waiting a minute, and then ending it so you can get used to sending yourself data before trying to take longer datasets. Once that longer dataset's been collected, I find it easiest just to email it to myself and put it in a directory that way. 
