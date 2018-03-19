/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define EPS 0.00001

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    	// Initialize the num of particles
	num_particles=100;
	// Create normal distribution based on first position
	normal_distribution<double> dist_x(x, std[0]);
 	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	// Generate particles with normal distribution with mean on GPS values.
	default_random_engine gen;
	for (int i = 0; i < num_particles; i++) {

	    Particle particle;
	    particle.id = i;
	    particle.x = dist_x(gen);
	    particle.y = dist_y(gen);
	    particle.theta = dist_theta(gen);
	    particle.weight = 1.0;

	    particles.push_back(particle);
	}

	// The filter is now initialized.
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	//vehicle motion
	if ( fabs(yaw_rate) < EPS ) { 
      	particles[i].x += velocity * delta_t * cos( theta );
      	particles[i].y += velocity * delta_t * sin( theta );
      	
    	} else {
      	particles[i].x += velocity / yaw_rate * ( sin( theta + yaw_rate * delta_t ) - sin( theta ) );
      	particles[i].y += velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate * delta_t ) );
      	particles[i].theta += yaw_rate * delta_t;
	}
	//add noise
	normal_distribution<double> dist_x(0, std[0]);
 	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);
	default_random_engine gen;
	particles[i].x += dist_x(gen);
    	particles[i].y += dist_y(gen);
	particles[i].theta += dist_theta(gen);
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0;i<observations.size();i++){
		double min_distance=numeric_limits<double>::max();
		int landmark_id;
		for(int j=0;j<predicted.size();j++){
			double dist_ij=dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if(dist_ij<min_distance){
				landmark_id=j;
				min_distance=dist_ij;
			}
		}
	observations[i].id=landmark_id;
	} 
	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double std_range=std_landmark[0];
	double std_bearing=std_landmark[1];
	
	for(int i=0;i<num_particles;i++){
		//particle position
		double x_p=particles[i].x;
		double y_p=particles[i].y;
		double theta_p=particles[i].theta;
		//find all landmarks within sensor_range
		vector<LandmarkObs> inRangeLandmarks;
		for(int j=0;j<map_landmarks.landmark_list.size();j++){
			//get landmark position and id
			double x_landmark=map_landmarks.landmark_list[j].x_f;
			double y_landmark=map_landmarks.landmark_list[j].y_f;
			int id=map_landmarks.landmark_list[j].id_i;
			double distance=dist(x_c,y_c,x_landmark,y_landmark);
			//landmark is within range if distance is not larger than sensor_range
			if(distance<=sensor_range){
				inRangeLandmarks.push_back(LandmarkObs{id, x_landmark, y_landmark});
			}
		}
		//convert landmark observation into map coordinate
		vector<LandmarkObs> observations_m;
		for(int j=0;j<observations.size();j++){
			double x_obs=observations[j].x;
			double y_obs=observations[j].y;
			double x_m=x_p+cos(theta_p)*x_obs-sin(theta_p)*y_obs;
			double y_m=y_p+sin(theta_p)*x_obs+cos(theta_p)*y_obs;
			observations_m.push_back(LandmarkObs{observations[j].id, x_m, y_m});
		}
		// find the landmark id for observations 
		dataAssociation(inRangeLandmarks, observations_m);
		// Calculate weights
		particles[i].weight=1.0;
		for(int i=0;i<observations_m.size();i++){
			double observation_x=observations_m[i].x;
			double observation_y=observations_m[i].y;
			double landmark_x, landmark_y;
			for(int j=0;j<inRangeLandmarks.size();j++){
				if(inRangeLandmarks[j].id==observations_m[i].id){
					landmark_x=inRangeLandmarks[j].x;
					landmark_y=inRangeLandmarks[j].y;
				}
			}
			double dx=observation_x-landmark_x;
			double dy=observation_y-landmark_y;
			double weight = ( 1/(2*M_PI*std_range*std_bearing)) * exp(-( dx*dx/(2*std_range*std_range) + (dy*dy/(2*std_bearing*std_bearing))));
			if(weight==0){
				particles[i].weight *=EPS;
			}else{
				particles[i].weight *=weight;
			}
	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
