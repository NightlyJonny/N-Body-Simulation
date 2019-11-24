#include "Drawer.h"

Drawer::Drawer(Simulation* sim) {
	
	this->particle = sim->getParticle();
	this->NParticle = sim->getNParticle();
}

void Drawer::drawFilledCircle(GLfloat x, GLfloat y, GLfloat radius) {
	const int triangleAmount = 40; //# of triangles used to draw circle

	glBegin(GL_TRIANGLE_FAN);
		glColor3f(1.0, 1.0, 1.0);
		glVertex2f(x, y); // center of circle
		for (int i = 0; i <= triangleAmount; i++) {
			glVertex2f(
				x + (radius * cos(i * 2*PI / triangleAmount)),
				y + (radius * sin(i * 2*PI / triangleAmount))
			);
		}
	glEnd();
}

void Drawer::drawSpaceship() {
	const GLfloat width = 0.3;
	const GLfloat length = 0.7;

	glBegin(GL_LINE_LOOP);
		glColor3f(1.0, 1.0, 1.0);
		glVertex2f(x, y);
		glVertex2f(x + width * cos(angle + 5*PI/8), y + width * sin(angle + 5*PI/8));
		glVertex2f(x + length * cos(angle), y + length * sin(angle));
		glVertex2f(x + width * cos(angle - 5*PI/8), y + width * sin(angle - 5*PI/8));
	glEnd();
}

void Drawer::setAsteroids(bool active) {

	asteroids = active;
}

void Drawer::rotateSpaceship(GLfloat da) {

	angle += da;
}

void Drawer::moveSpaceship(GLfloat ds) {

	x += ds * cos(angle);
	y += ds * sin(angle);
}

void Drawer::shoot() {

	glBegin(GL_LINES);
		glColor3f(1.0, 0, 0);
		glVertex2f(x, y);
		glVertex2f(x + 100 * cos(angle), y + 100 * sin(angle));
	glEnd();
}

void Drawer::draw_scene() {
	for (int i = 0; i < NParticle; i++) {
		Vector2 pos = particle[i].position;
		drawFilledCircle(pos.getx(), pos.gety(), particle[i].radius);
	}
	if (asteroids) drawSpaceship();
	if (shooting) shoot();
}
