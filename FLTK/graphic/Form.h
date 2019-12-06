#ifndef FORM_HPP
#define FORM_HPP

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Multiline_Output.H>
#include <string>
#include "Frame.h"

class Form{
private:
	static Fl_Window* form;
	static Frame* scene;

	Fl_Output* timeText;
	Fl_Output* debugText;

	int floatToInt(float num, int afterComma) { return round(num * pow(10, afterComma)); }
	void CreateMyWindow();
public:
	static Simulation* sim;
	Form(Simulation*, bool);
	
	void updateTime(float nextTime);
	void updateDebug(string text);

	Frame* getFrame();
};

#endif // FORM_HPP
