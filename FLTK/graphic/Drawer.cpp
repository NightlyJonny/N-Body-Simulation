#include "Drawer.h"

Drawer::Drawer(Simulation* sim) {
	
	this->particle = sim->getParticle();
	this->NParticle = sim->getNParticle();
}

void Drawer::drawFilledCircle(GLfloat x, GLfloat y, GLfloat radius) {
	int i;
	int triangleAmount = 40; //# of triangles used to draw circle

	//GLfloat radius = 0.8f; //radius
	GLfloat twicePi = 2.0f * 3.14159265359;

	glBegin(GL_TRIANGLE_FAN);
		glColor3f(1.0, 1.0, 1.0);
		glVertex2f(x, y); // center of circle
		for (i = 0; i <= triangleAmount; i++) {
			glVertex2f(
				x + (radius * cos(i * twicePi / triangleAmount)),
				y + (radius * sin(i * twicePi / triangleAmount))
			);
		}
		glEnd();
}

// ********************************************************************************************************
void Drawer::draw_scene() {

	for (int i = 0; i < NParticle; i++) {
		Vector2 pos = particle[i].position;
		drawFilledCircle(pos.getx(), pos.gety(), particle[i].radius);	// draw the torus
	}
}
