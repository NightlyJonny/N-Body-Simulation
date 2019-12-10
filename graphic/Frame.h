#ifndef FRAME_HPP
#define FRAME_HPP

#define SCREEN_HEIGHT 600
#define SCREEN_WIDTH 600

#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/gl.h>
#include <GL/glu.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iterator> 
#include <map>
#include <vector>

#include "Drawer.h"
#include "../Simulation.h"

class Frame : public Fl_Gl_Window {

private:
	Drawer* drawer = nullptr;
	int* frames;
	int keycode;
	bool asteroids = false;
	
	int mouseInitPos[2]; //This store the initial coordinates of the mouse usefu; for rotate the camera
	float angleStep = 0.02; //How much the angle is augmented at every step
	map<int, bool> buttDown; //Indicates which buttons are pressed
	bool vision3d = false; //Is the 3d vision enabled?
	float angle; //Angle of rotation of the camera

	bool rightDown = false; //store if the right mouse button is pressed

	double zoom = 0.2, xshift = 0, yshift = 0, zshift = 0;

	int handle(int);
	void draw();
	void timer(int);
	void init();

public:
	
	Frame(int x, int y, int w, int h, const char* l = 0);
	void setSimulation(Simulation* sim);
};


#endif // FRAME_1_HPP