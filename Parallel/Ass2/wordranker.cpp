#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <string>
#include <sstream>
#include <array>
#include <omp.h>
#include <vector>
#include <unordered_map>

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}


using namespace std;

int main(int argc, char *argv[]) {

  ios::sync_with_stdio(false);//For faster IO

  unordered_map<string, int> index; 

  string files = exec("find ./files -type f");
  stringstream ss(files);
  vector<string> filelist;
  string f;
  while(getline(ss, f, '\n')){
  	filelist.push_back(f);
  }

  omp_set_num_threads(4);

  #pragma omp parallel for schedule(static)
  for(int i=0; i < filelist.size(); ++i){
	unordered_map<string, int> localindex;
	string s;
	unordered_map<string, int>::iterator it, it2;
	ifstream document(filelist[i]);
	while(document >> s){
		it = localindex.find(s);
		if (it == localindex.end()){
			localindex[s] = 1;
		}
		else ++(localindex[s]);
	}
	document.close();
	#pragma omp critical
	{
	  for( it = localindex.begin(); it != localindex.end(); ++it){
	  	it2 = index.find(it->first);
		if (it2 == index.end()){
			index[it->first] = it->second;
		}
		else index[it->first] += it->second;
	  }
	}
  }
  
  cout << index.size() << endl;

  for(unordered_map<string, int>::iterator it = index.begin(); it != index.end(); ++it){
  	cout << it->first << " " << it->second << endl;
  } 

}
