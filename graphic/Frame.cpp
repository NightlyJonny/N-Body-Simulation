#include "Frame.h"
#include "Form.h"
#include <string>

#define MINZOOM 0.001
#define MAXZOOM 0.9
#define SHIFTSTEP 1
#define ROTATIONSPEED 0.1
#define SPEED 0.3

enum Button { A=97, S=115, D=100, W=119, X=120, Q=113, Z=122, SPACE=32, LEFT=65363, RIGHT=65361, UP=65362, DOWN=65364};

extern Form* form;

Frame::Frame(int x, int y, int w, int h, const char* l) : Fl_Gl_Window(x, y, w, h, l) {
	drawer = new Drawer();

	vector<int> butt = {A, S, D, W, X, Q, Z, LEFT, RIGHT, UP, DOWN};

	for(int n : butt){

		buttDown.insert(pair<int, bool>(n, false)); 
	}

	pitch = 0;
	yaw = 0;
	angle = 0;

};

void Frame::init(void) {

	gl_font(FL_HELVETICA_BOLD, 16);
}

void Frame::setSimulation(Simulation* sim) {
	
	drawer->setSimulation(sim);
	angle = 0;
	xshift = 0;
	yshift= 0;
	zshift = 0;
	pitch = 0;
	yaw = 0;
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
		glViewport(0, 0, pixel_w(), pixel_h());                      // Make our viewport the whole window
		glMatrixMode(GL_PROJECTION);                           // Select The Projection Matrix
		glLoadIdentity();                                      // Reset The Projection Matrix
		gluPerspective(45.0f, ((float)pixel_w()) / pixel_h(), 0.5, 150.0f);
		glMatrixMode(GL_MODELVIEW);                            // Select The Modelview Matrix
		glLoadIdentity();                                      // Reset The Modelview Matrix
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    // Clear The Screen And The Depth Buffer
		glLoadIdentity();                                      // Reset The View
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);

		init();
		valid(1);
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(!vision3d){
		//This move the scene is the mouse right button is pressed
		if(rightDown){
			float degree = -atan2(pixel_h()/2 - Fl::event_y(), pixel_w()/2 - Fl::event_x());
			float ray = pow(pow(pixel_h()/2 - Fl::event_y(), 2) + pow(pixel_w()/2 - Fl::event_x(), 2), 1./2);
			xshift += (cos(degree) * ray / 400) * cos(pitch);
			yshift += (sin(degree) * ray / 400);
			zshift += (cos(degree) * ray / 400) * sin(pitch);
		}
	}else{

		if(rightDown){
			float deltax = pixel_w()/2 - Fl::event_x();
			float deltay = pixel_h()/2 - Fl::event_y();
			
			if(deltax != 0 && deltay != 0){

				float mod = sqrt(pow(deltax, 2) + pow(deltay, 2));
				deltax /= mod;
				deltay /= mod;

				pitch -= deltax / 100;
				yaw -= deltay / 100;

				if(abs(yaw * 180 / M_PI) > angleLimit){
					yaw = copysign(angleLimit, yaw) / 180 * M_PI;
				}
			}
		}

		float step = 0.2;
		for (const auto &pair : buttDown) {
			if(pair.second == true){

				if(pair.first == A) xshift += step*cos(pitch)*cos(yaw), zshift += step*sin(pitch)*cos(yaw), yshift += step*sin(yaw);
				if(pair.first == D) xshift -= step*cos(pitch)*cos(yaw), zshift -= step*sin(pitch)*cos(yaw), yshift -= step*sin(yaw);
				if(pair.first == S) xshift += step*sin(pitch)*cos(yaw), zshift -= step*cos(pitch)*cos(yaw), yshift -= step*sin(yaw);
				if(pair.first == W) xshift -= step*sin(pitch)*cos(yaw), zshift += step*cos(pitch)*cos(yaw), yshift += step*sin(yaw);
				if(pair.first == Z) yshift += step;
				if(pair.first == Q) yshift -= step;
			}
		}
	}

	glPushMatrix();
		glScalef(zoom, zoom, zoom);
		glRotatef(pitch * 180 / M_PI, 0, 1, 0);
		glRotatef(yaw * 180 / M_PI, cos(pitch), 0, sin(pitch));
		
		glTranslatef(xshift, yshift, zshift - 40);
		drawer->draw_scene();

		form->updateTime(*frames * 1. / FRAMERATE);
		form->updateDebug(form->sim->debugText);
		
	glPopMatrix();

}

int Frame::handle(int event) {

	switch (event)
	{

		case FL_MOUSEWHEEL:
			if ((zoom - Fl::event_dy() * 0.01 < MAXZOOM) && (zoom - Fl::event_dy() * 0.01 > MINZOOM)) zoom -= Fl::event_dy() * 0.01;
			cout << "miao" << endl;
			break;
		
		case FL_KEYDOWN:

			keycode = Fl::event_key();
			if(!vision3d){
				if (keycode == A) drawer->setAsteroids(asteroids = !asteroids); // "A"
				if (asteroids) {
					if (keycode == LEFT) drawer->rotateSpaceship(-ROTATIONSPEED);// Left arrow
					if (keycode == RIGHT) drawer->rotateSpaceship(ROTATIONSPEED); // Right arrow
					if (keycode == UP) drawer->moveSpaceship(SPEED); // Up arrow
					if (keycode == SPACE) drawer->shooting = true; // Spacebar
					else drawer->shooting = false;
				}
				else {
					drawer->shooting = false;
					if (keycode == LEFT) xshift -= SHIFTSTEP; // Left arrow
					if (keycode == RIGHT) xshift += SHIFTSTEP; // Right arrow
					if (keycode == UP) yshift -= SHIFTSTEP; // Up arrow
					if (keycode == DOWN) yshift += SHIFTSTEP; // Down arrow
				}
			}
			
			if(!asteroids)
				if (keycode == SPACE) form->sim->togglePause(); // Spacebar

			if(keycode == X){ //Button X
				vision3d = !vision3d;
				if(vision3d)
					form->updateInfo("3D Vision: on");
				else
					form->updateInfo("3D Vision: off");
			}
			
			buttDown[keycode] = true; //Set the button on "Pressed" status

			break;

		case FL_KEYUP:
			keycode = Fl::event_key();
			buttDown[keycode] = false;

			break;
		case FL_PUSH:
			if(Fl::event_button() == FL_RIGHT_MOUSE){
				this->rightDown = true;
			}
			break;
		case FL_RELEASE:
			if(Fl::event_button() == FL_RIGHT_MOUSE)
				this->rightDown = false;
			break;
		default:
			break;

		drawer->shooting = false;
		redraw();
	}
	
	return 1;
}