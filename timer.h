
/* $Id: timer.h 93 2008-04-11 01:11:38Z wangsl $ */

#ifndef TIMER_H
#define TIMET_H

#include <iostream>
using namespace std;
#include <unistd.h>
#include <ctime> 
#include <sys/times.h>

class Timer
{
public:
  Timer()
  { 
    times(&start_time);
    ticks_per_second = sysconf(_SC_CLK_TCK);
  }

  ~Timer() { }
  
  void reset() { times(&start_time); }
  
  double time() const
  { 
    struct tms end_time;
    times(&end_time);
    clock_t diff = end_time.tms_utime - start_time.tms_utime;
    double seconds = ((double) diff)/ticks_per_second;
    return seconds;
  }
  
  static void print_time(const double &sec)
  {
    const streamsize default_precision = cout.precision();

    cout.precision(2);

    if(sec < 60) 
      cout << " " << fixed << sec << " secs";
    else 
      if(sec < 3600)  
	cout << " " << int(sec/60) << " mins, " << sec-int(sec/60)*60 << " secs";
      else 
        if(sec < 86400) 
	  cout << " " << int(sec/3600) << " hrs, " 
	       << (sec-int(sec/3600)*3600)/60.0 << " mins";
        else   
          cout << " " << int(sec/86400) << " days, " 
	       << (sec-int(sec/86400)*86400)/3600.0 << " hrs";
    cout << endl;

    cout.precision(default_precision);
  }
  
private:
  struct tms start_time;
  int ticks_per_second;
};

void reset_timer();
double time();

#endif /* TIMER_H */
