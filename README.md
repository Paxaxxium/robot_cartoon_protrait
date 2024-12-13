# robot_cartoon_portrait

## Dependencies
### Python Libraries
`python -m pip install -r requirements.txt`

## Instructions
### Creating the trajectories
Copy absolute path of image into path object in experiment.py then run using  `experiement.py `
### Drawing the portrait
Drawing the portrait: `python integrated_client_fb_page.py`

## Output
* Print out the segementation colors used by pyfacer
* Print out the dimension of the input image
* 13 different trajectories(stored in CSV files) used to draw the image
  * 2 for eyes(each)
  * 2 for eyebrows(each)
  * 2 for lips
  * 1 for hair
  * 1 for face
* Convert pixel trajerctories into world frame
* Joint angles of robot from inverse kinematics
* Robot draws portrait using robot control and force control
### Citation
part of code are attributed to `https://github.com/rpiRobotics/robotic_portrait.git` from Rensselear Polytechnic Institute
