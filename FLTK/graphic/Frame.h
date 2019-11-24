#ifndef FRAME_HPP
#define FRAME_HPP

#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/gl.h>
#include <GL/glu.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "Drawer.h"
#include "../Simulation.h"

class Frame : public Fl_Gl_Window {

private:
	Drawer* drawer = nullptr;
	int* frames;
	int keycode;
	bool asteroids = false;

	int handle(int);
	void draw();
	void timer(int);
	void init();

public:
	double ruotaX, ruotaY, ruotaZ, zoom = 0.2, xshift = 0, yshift = 0;
	Frame(int x, int y, int w, int h, const char* l = 0) : Fl_Gl_Window(x, y, w, h, l) {};
	void setSimulation(Simulation* sim);
};


#endif // FRAME_1_HPP