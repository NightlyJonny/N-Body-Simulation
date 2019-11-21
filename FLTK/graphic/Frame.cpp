#include "Frame.h"
#include "Form.h"
#include <string>

#define MINZOOM 0.01
#define MAXZOOM 0.75
#define SHIFTSTEP 1

extern Form* form;

void Frame::init(void) {

	gl_font(FL_HELVETICA_BOLD, 16);
}

void Frame::setSimulation(Simulation* sim) {
	drawer = new Drawer(sim);
	frames = sim->getFrameRef();
}

void Frame::draw() {
	if (!valid()) {
		GLfloat amb_light[] = { 0.1, 0.1, 0.1, 1.0 };
		GLfloat diffuse[] = { 0.6, 0.6, 0.6, 1 };
		GLfloat specular[] = { 0.7, 0.7, 0.3, 1 };
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, amb_light);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
		glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
		glEnable(GL_LIGHT0);
		glEnable(GL_COLOR_MATERIAL);
		glShadeModel(GL_SMOOTH);
		glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
		glDepthFunc(GL_LEQUAL);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glClearColor(0.0, 0.0, 0.0, 1.0);

		glClearColor(0.0, 0.0, 0.0, 1);                        // Turn the background color black
		glViewport(0, 0, w(), h());                      // Make our viewport the whole window
		glMatrixMode(GL_PROJECTION);                           // Select The Projection Matrix
		glLoadIdentity();                                      // Reset The Projection Matrix
		gluPerspective(45.0f, w() / h(), 1, 150.0f);
		glMatrixMode(GL_MODELVIEW);                            // Select The Modelview Matrix
		glLoadIdentity();                                      // Reset The Modelview Matrix
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    // Clear The Screen And The Depth Buffer
		glLoadIdentity();                                      // Reset The View
		gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0);        // Position - View  - Up Vector
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);

		init();
		valid(1);
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();
		glScalef(zoom, zoom, zoom);
		glTranslatef(xshift, yshift, 0);
		drawer->draw_scene();
		form->updateTime(*frames * 1. / FRAMERATE);
	glPopMatrix();
}

int Frame::handle(int event) {
	switch (event)
	{
		case FL_MOUSEWHEEL:
			if ((zoom - Fl::event_dy() * 0.01 < MAXZOOM) && (zoom - Fl::event_dy() * 0.01 > MINZOOM)) zoom -= Fl::event_dy() * 0.01;
			break;
		case FL_KEYDOWN:
			keycode = Fl::event_key();
			if (keycode == 32) form->sim->togglePause(); // Spacebar
			if (keycode == 65363) xshift -= SHIFTSTEP; // Left arrow
			if (keycode == 65361) xshift += SHIFTSTEP; // Right arrow
			if (keycode == 65362) yshift -= SHIFTSTEP; // Up arrow
			if (keycode == 65364) yshift += SHIFTSTEP; // Down arrow
			return 1;
		default:
			break;
		redraw();
	}
	
	return 1;
}