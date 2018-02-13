#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <string>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <limits>
#include "hungarian.hpp"

using namespace std;
typedef unsigned int ll;

/*0. data structure speedups: represent as

vector < short int> req; req [C*TT*3 + T*3+data]; req.end means unassigned
// req[0] does not exist so shchedule[x]==0 means unassigned
where data = enum{lessons,maxperday,doubllessons}

vector < int > schedule; schedule [C*25 + tslot] = req_pos

vector < bool > unavbl; unavbl [T*25 + tslot]
unavbl [ x ]==1 means teacher cannot teach at that time*/

unsigned int nclss, ntchs, ndys, nprds, ndps;
vector<unsigned int> teacher_timeslots, assignments, teacher_wd, double_lessons;

// components calculation
vector<unsigned int> component;
vector< bool > comp_graph, comp_visited;

struct req{ unsigned short int lsns, mxpd, dbls;};
vector<req> reqs;
vector<bool> unavbls;

struct Solution{
  vector<vector<req>::iterator>  schd; //schedule
  //hard constraint, soft constraint, total
  volatile unsigned int hcs, scs, total;
  //max_lessons_day, conflicts_teachers, availabilities
  volatile unsigned int maleda, cote, avai;
  //minimum_double_lessons, idle_times, teacher_compactness
  volatile unsigned int midole, idti, teco;

  /*
  Solution()
  {
    hcs = scs = total = 0;
    maleda = cote = avai = 0;
    midole = idti = teco = 0;
  }//*/

  void UpdateSize()
  {
    schd.assign(nclss*ndps, reqs.end());
  }
};


bool isNum(char a)
{
  if(a == '0' || a == '1' || a == '2' || a == '3' || a == '4' || a == '5' || a == '6' || a == '7' || a == '8' || a == '9') return true;
  return false;
}

// mide el tiempo (en milisegundos) transcurrido
// entre elapsed(true) y elapsed()
inline int elapsed(bool reset = false)
{
	static clock_t start = clock();
	if ( reset ) start = clock();
	return (1000.00*double(clock()-start))/double(CLOCKS_PER_SEC);
}

inline void readXMLLine(string entity, vector<unsigned int>& values)
{
  string num;
  unsigned int val;
  for(unsigned int i = 0; i < entity.size(); i++)
  {
    if(isNum(entity[i]))
    {
      num += entity[i];
    }
    else
    {
      if(num.size() > 0)
      {
        istringstream (num) >> val;
        values.push_back(val);
        num = "";
      }
    }
  }
}


void readXML(string filename)
{
  vector<unsigned int> values;
  string word;
  ifstream file(filename);

  for(unsigned int i = 0; i < 4; i++)
  {
    file >> word;
  }
  for(unsigned int i = 0; i < 4; i++)
  {
    //values.clear();
    getline(file,word);
    //cout<<word<<endl;
    readXMLLine(word,values);
    //for(unsigned int j = 0; j < values.size(); j++) cout<<values[j]<<endl;
  }

  nclss = values[1] + 1;
  ntchs = values[3] + 1;
  ndys  = values[5] + 1;
  nprds = values[7] + 1;
  ndps = ndys * nprds;

  for(unsigned int i = 0; i < 2; i++) file >> word;

  reqs.assign(nclss * ntchs, {0,0,0} );

  while(file >> word && word != "</requirements>")
  {
    values.clear();
    getline(file,word);

    readXMLLine(word,values);
    //for(unsigned int j = 0; j < values.size(); j++) cout<<values[j]<<endl;

    //  reqs [ ( theClass * ntchs + theTeacher ) ] // pos 0 means unasigned
    auto rq = reqs.begin() + values[0] * ntchs  + values[1];

    (*rq).lsns = values[2];     // lessons
    (*rq).mxpd = values[3]; // max per day
    (*rq).dbls = values[4]; // double lessons

  }

  file >> word;

  unavbls.assign(ntchs * ndps,0);
  while(file >> word && word != "</teacherunavailabilities>")
  {
    values.clear();
    getline(file,word);
    //cout<<word<<endl;
    readXMLLine(word,values);
    //for(unsigned int j = 0; j < values.size(); j++) cout<<values[j]<<endl;

    // unavbls [ theTeacher * ndps + theDay * nprds + thePeriod] = 1 // he cant that period
    *(unavbls.begin() + values[0] * ndps + values[1] * nprds + values[2]) = 1;
  }
  file.close();
  teacher_timeslots.reserve(ntchs * ndps);
  assignments.reserve(reqs.size() * ndys);
  teacher_wd.reserve(ntchs * ndys);
  double_lessons.reserve(reqs.size());
  component.reserve(nclss);
  comp_graph.reserve(nclss*nclss);
  comp_visited.reserve(nclss);
}

void fillTeachers(Solution & sol)
{
  fill(teacher_timeslots.begin(), teacher_timeslots.end() , 0);
  fill(teacher_wd.begin(), teacher_wd.end() , 0);
  for( auto rq = sol.schd.begin(); rq<sol.schd.end(); )
  {
    for(unsigned int j = 0; j < ndps; j++)
    {
      //cout<<j<<endl;
      if ( *rq < reqs.end() ) // req exists?
      {
        //cout<<t_name_idx[solution->m_schedule[j][i]->m_teacher]<<" "<<r_name_idx[solution->m_schedule_rooms[j][i]->m_name]<<endl;
        //cout<<"teacher id en pos: "<<i<<" "<<j<<" ";
        //teacher_idx = solution->m_schedule[i][j]->m_teacher;
        //cout<<teacher_idx<<endl;
        teacher_timeslots [ ( unsigned ( *rq - reqs.begin() ) % ntchs ) * ndps + j]++;
        teacher_wd [ ( unsigned ( *rq - reqs.begin() ) % ntchs ) * ndys + j / nprds] = 1;
      }
      ++rq;
    }
  }
}

void fillAssignments( Solution &sol )
{
  fill(assignments.begin(), assignments.end() , 0);

  for( auto rq = sol.schd.begin(); rq<sol.schd.end(); )
  {
    for(unsigned int j = 0; j < ndps; j++)
    {
      //cout<<j<<endl;
      if ( *rq < reqs.end() ) // req exists?
      {
        assignments [ unsigned ( *rq - reqs.begin() ) * ndys + j / nprds]++;
      }
      ++rq;
    }
  }
}

void fillDoubleLessons( Solution &sol )
{
  fill(double_lessons.begin(), double_lessons.end() , 0);
  auto cdl = double_lessons.begin();
  for( auto rq = sol.schd.begin(); rq<sol.schd.end(); )
  {
    for(unsigned int d = ndys; d; --d)
    {
      auto lrq = *rq; ++rq;
      for(unsigned int j = nprds - 1; j; --j)
      {
        if ( *rq == lrq )
        double_lessons [ unsigned ( lrq - reqs.begin() ) ] ++;
        lrq = *(rq++);
      }
    }
  }
}

unsigned int conflicts(Solution &sol)
{
  fillTeachers(sol);
  unsigned int sum = 0;
  for ( auto & c : teacher_timeslots ) if ( c > 1 ) sum += c - 1;
  return sum;
}

unsigned int teacherCompactness(Solution &sol)
{
  // Ensure conflicts was called before this, due to fillTeachers.
  unsigned int sum = 0;
  for ( auto & c : teacher_wd ) sum += c;
  return sum;
}

unsigned int availabilities(Solution &sol)
{
  unsigned int sum = 0;
  for( auto rq = sol.schd.begin(); rq<sol.schd.end(); )
  {
    for(unsigned int j = 0; j < ndps; j++)
    {
      if ( *rq < reqs.end() ) // req exists?
      if ( unavbls [ ( unsigned ( *rq - reqs.begin() ) % ntchs ) * ndps + j ] )
      ++sum;
      ++rq;
    }
  }
  return sum;
}

unsigned int maxLessonsDay(Solution &sol)
{
  fillAssignments(sol);
  unsigned int sum = 0;
  auto as = assignments.begin();
  for ( auto & rq : reqs )
  {
    for (unsigned int d = ndys; d; --d)
    {
      if ( *as > rq.mxpd ) sum += *as - rq.mxpd;
      ++as;
    }
  }
  return sum;
}

unsigned int minDoubleLessons(Solution &sol)
{
  fillDoubleLessons(sol);
  unsigned int sum = 0;
  auto rq = reqs.begin();
  for(auto &a : double_lessons)
  {
    if ( a < (*rq).dbls ) sum += (*rq).dbls - a;
    ++rq;
  }
  return sum;
}

unsigned int countIdleTimes(Solution &sol)
{
  // Ensure conflicts was called before this, due to fillTeachers.
  static unsigned short int tbl [32] = {0,0,0,0, 0,1,0,0, 0,2,1,1, 0,1,0,0, 0,3,2,2, 1,2,1,1, 0,2,1,1, 0,1,0,0};
  unsigned int sum = 0;
  for(auto it = teacher_timeslots.begin(); it < teacher_timeslots.end(); )
  {
    for(unsigned int d = ndys; d; --d)
    {
      short int p = 0;
      for(unsigned int j = nprds; j; --j)
      {
        p<<=1;
        if ( *it ) ++p;
        ++it;
      }
      sum += tbl [p];
    }
  }
  return sum;
}

unsigned int getHardConstraints(Solution &sol)
{
  //cout<<"hard constraints"<<endl;
  unsigned int mld = maxLessonsDay(sol);
  //cout<<"max lessons day: "<<mld<<endl;
  unsigned int c = conflicts(sol);
  //cout<<"conflicts teachers: "<<c<<endl;
  unsigned int a = availabilities(sol);
  //cout<<"availabilities: "<<a<<endl<<endl;

  mld *= 10000; c *= 100000; a *= 100000;
  sol.maleda = mld;
  sol.cote = c;
  sol.avai = a;
  return c+ a+ mld;

}

unsigned int getSoftConstraints(Solution &sol)
{
  //cout<<"soft constraints: "<<endl;
  unsigned int tc = teacherCompactness(sol);
  //cout<<"teacher compactness: "<<tc<<endl;
  unsigned int mdl = minDoubleLessons(sol);
  //cout<<"minimum double lessons: "<<mdl<<endl;
  unsigned int it = countIdleTimes(sol);
  //cout<<"idle times: "<<it<<endl<<endl;

  tc *= 9; mdl *= 1; it *= 3;

  sol.midole = mdl;
  sol.teco = tc;
  sol.idti = it;
  return mdl+it+tc;
}

void fitness(Solution &sol)
{
  sol.hcs = getHardConstraints(sol);
  sol.scs = getSoftConstraints(sol);
  sol.total = sol.hcs + sol.scs;
}

void generateSolution(Solution & sol)
{
  vector<unsigned short int> avaiDp;
  int lectures = 0, cl = -1, last_cl = cl, total_lec = 0;
  int random_ts;
  //cout<<num_q<<" "<<num_ts<<endl;
  sol.UpdateSize();
  avaiDp.assign(ndps,0);
  iota(avaiDp.begin(),avaiDp.end(),0);
  auto rq = reqs.begin();
  for(unsigned int c = 0; c < nclss; ++c)
  {
    random_shuffle(avaiDp.begin(),avaiDp.end());
    auto adp = avaiDp.begin();
    for(unsigned int t = 0; t < ntchs; ++t)
    {
      lectures = (*rq).lsns;

      //cout<<"Clase: "<<q<<" "<<lectures<<endl;
      while(lectures > 0)
      {
        //cout<<"eligiendo random de: "<<solution.m_available_ts[q].size()<<endl;
        //cout<<"random elegido: "<<random_ts<<endl;
        sol.schd[c * ndps + *adp] = rq;
        adp++;

        //cout<<"end"<<endl;
        lectures--;
      }

      rq++;
    }
  }
  fitness(sol);
}

inline void TQMove(Solution & sol, unsigned int ti, unsigned int tj)
{
  comp_graph.assign(nclss*nclss, 0);
  for ( auto iti = sol.schd.begin() + ti, itj = sol.schd.begin() + tj; iti < sol.schd.end(); iti+=ndps, itj+=ndps )
  {
    if ( *iti != *itj )
    {
      if ( *itj == reqs.end() ) // *iti exists
      {
        unsigned int tchr = unsigned ( *iti - reqs.begin() ) % ntchs;
        unsigned int clss = ( unsigned ( iti - sol.schd.begin() ) / ndps );
        for ( auto jtj = itj + ndps; jtj < sol.schd.end(); jtj+=ndps )
          if ( *jtj != reqs.end() )
            if ( tchr == unsigned ( *jtj - reqs.begin() ) % ntchs )
              comp_graph[ clss * nclss + unsigned ( jtj - sol.schd.begin() ) / ndps ] =
              comp_graph[ ( unsigned ( jtj - sol.schd.begin() ) / ndps ) * nclss + clss ] = 1;
      }
      if ( *iti == reqs.end() ) // *itj exists
      {
        unsigned int tchr = unsigned ( *itj - reqs.begin() ) % ntchs;
        unsigned int clss = unsigned ( itj - sol.schd.begin() ) / ndps;
        for ( auto jti = iti + ndps; jti < sol.schd.end(); jti+=ndps )
          if ( *jti != reqs.end() )
            if ( tchr == unsigned ( *jti - reqs.begin() ) % ntchs )
              comp_graph[ clss * nclss + unsigned ( jti - sol.schd.begin() ) / ndps ] =
              comp_graph[ ( unsigned ( jti - sol.schd.begin() ) / ndps ) * nclss + clss ] = 1;
      }
    }
  }
}

void DFS(unsigned int rt_clss)
{
  component.clear();
  vector < unsigned int > stck;
  stck.reserve(nclss);
  stck.push_back(rt_clss);
  while(!stck.empty())
  {
    unsigned int clss = stck.back();
    stck.pop_back();
    comp_visited[clss] = 1;
    component.push_back(clss);
    unsigned int d = 0;
    for ( auto it = comp_graph.begin() + clss * nclss, _it = it + nclss; it < _it; ++it, ++d)
      if (*it) if (! comp_visited[d] ) stck.push_back(d);
  }
}

inline void applySwap(Solution & sol, unsigned int ti, unsigned int tj)
{
  for (auto & c : component)
  {
    auto t = sol.schd[c * ndps + ti];
    sol.schd[c * ndps + ti] = sol.schd[c * ndps + tj];
    sol.schd[c * ndps + tj] = t;
  }
}

void perturbation(Solution & sol)
{
  unsigned int ti = unsigned( rand() ) % ndps;
  unsigned int tj = unsigned( rand() ) % ndps;
  while(ti == tj) tj = unsigned( rand() ) % ndps;
  TQMove (sol, ti, tj);
  DFS( unsigned ( rand() ) % nclss); // get components with a depth first search from a random class as root
  applySwap(sol, ti, tj);
}

void localSearchTQ(Solution & bsol)
{
  Solution csol = bsol;
  comp_visited.assign(nclss, 0);
  unsigned int oldbest;
  do
  {
    oldbest = bsol.total;
    for(unsigned int i = 0; i < ndps; i++)
    {
      for(unsigned int j = 0; j < ndps; j++)
      {
        if(i == j) continue;
        TQMove(csol, i, j);

        for(unsigned int k = 0; k < nclss; k++)
        {
          if (comp_visited[k]) continue;
          DFS(k); // get components with a depth first search from a class as root
          applySwap(csol, i, j);
          fitness(csol);
          if(csol.total <= bsol.total) bsol = csol;
        }
      }
    }
  }
  while ( bsol.total < oldbest);
}

void iteratedLocalSearchTQ(Solution & bsol, int stop_condition)
{
  unsigned int not_improved = 0;
  stop_condition*=1000; // to milliseconds for precision
  elapsed(true);
  generateSolution(bsol);
  fitness(bsol);
  localSearchTQ(bsol);
  Solution csol = bsol;
  cout<<endl<< ">> " <<bsol.total<<endl;

  while ( elapsed() < stop_condition )
  {
    //cout<<i<<endl;
    //cout<<"ini perturbation"<<endl;
    perturbation(csol);
    //cout<<"perturbed"<<endl;
    fitness(csol);
    localSearchTQ(csol);

    if ( bsol.total > csol.total )
      not_improved = 0;
    else
      not_improved++;

    if ( bsol.total >= csol.total )
      bsol = csol;

    if(not_improved >= 3)
    {
      csol = bsol;
      not_improved = 0;
    }
  }
}


#ifdef nodefinido

struct Solution;


clock_t start;

void myFill(unsigned int** &a, unsigned int n,unsigned int m, int val)
{

  for(unsigned int i = 0; i < n; i++)
  {
    for(unsigned int j = 0; j < m; j++)
    {
      a[i][j] = val;
    }
  }
  return;
}

inline bool isIn(vector<string> &a, string word)
{
  for(unsigned int i = 0; i < a.size(); i++)
  {
    if(a[i] == word) return true;
  }
  return false;
}

struct Requirement
{
  unsigned int m_double_lessons, m_max_per_day, m_lessons, m_teacher, m_class, m_id;
};

struct Teacher_unavailability
{
  unsigned int m_teacher, m_period, m_day;
};

struct High_school
{
  unsigned int m_classes, m_teachers, m_days, m_periods;
  vector<Requirement*> requirements;
  vector<Teacher_unavailability*> teacher_unavailabilities;
};


inline void getComponents(vector<vector<unsigned int> > &graph, vector<vector<unsigned int> >& components)
{
  //components.clear();
  vector<unsigned int> component;
  vector<bool> visited(graph.size());
  for(unsigned int i = 0; i < graph.size(); i++)
  {
    component.clear();
    if(!visited[i])
    {
      DFS(graph,i,component,visited);
      components.push_back(component);
    }
  }
  return;
}

inline sol_format applySwap(sol_format solution, vector<unsigned int> &component, int ti, int tj)
{
  for(unsigned int i = 0; i < component.size(); i++)
  {
    swap(solution.m_schedule[component[i]][ti], solution.m_schedule[component[i]][tj]);
  }
  return solution;
}




inline bool isBetter(sol_format* solution1, sol_format* solution2)
{
  /*if(solution1->m_hc_score < solution2->m_hc_score) return true;
  else if(solution1->m_hc_score == solution2->m_hc_score)
  {
  return solution1->m_sf_score < solution2->m_sf_score;
}
return false;*/
return (solution1->m_hc_score+solution1->m_sf_score) < (solution2->m_hc_score+solution2->m_sf_score);
}

inline void chooseRandomSubset(vector<unsigned int>& empty_subset, unsigned int max_num, unsigned int how_many)
{
  vector<unsigned int> full_set(max_num,0);
  unsigned int rand_idx = 0;
  for(unsigned int i = 0; i < max_num; i++) full_set[i] = i;

  for(unsigned int i = 0; i < how_many; i++)
  {
    rand_idx = rand() % full_set.size();
    empty_subset.push_back(full_set[rand_idx]);
    full_set.erase(full_set.begin()+rand_idx);
  }
  return;
}

inline void getRequirements(sol_format &solution, vector<unsigned int>& ts_subset, vector<Requirement*>& empty_subset, unsigned int c)
{
  for(unsigned int i = 0; i < ts_subset.size(); i++)
  {
    empty_subset.push_back(solution.m_schedule[c][ts_subset[i]]);
  }
}

vector<vector<ll> > calculateCostMatrix(High_school* &high_school,sol_format solution, vector<unsigned int>& ts_subset, vector<Requirement*>& r, unsigned int c)
{
  //cout<<"calculateCost ini: "<<ts_subset.size()<<endl;
  unsigned int ts_subset_tam = ts_subset.size();

  vector<vector<ll> >cost_matrix(ts_subset_tam,vector<ll>(ts_subset_tam,0));

  for(unsigned int i = 0; i < ts_subset_tam; i++)
  {
    solution.m_schedule[c][ts_subset[i]] = NULL;
  }

  //fitness(high_school, &solution);

  for(unsigned int i = 0; i < ts_subset_tam; i++)
  {
    for(unsigned int j = 0; j < ts_subset_tam; j++)
    {
      solution.m_schedule[c][ts_subset[j]] = r[i];
      fitness(high_school, &solution);
      cost_matrix[i][j] = solution.m_hc_score + solution.m_sf_score;
      solution.m_schedule[c][ts_subset[j]] = NULL;
      fitness(high_school, &solution);
    }
  }
  return cost_matrix;
}

void updateSolution(High_school* &high_school, sol_format &solution, const vector<vector<ll> > &assignment_matrix, vector<unsigned int>& ts_subset, vector<Requirement*>& r, unsigned int c)
{
  //cout<<"updating solution"<<endl;
  for(unsigned int i = 0; i < assignment_matrix.size(); i++)
  {
    for(unsigned int j = 0; j < assignment_matrix[i].size(); j++)
    {
      if(assignment_matrix[i][j] == 1)
      {
        //cout<<r[i]->m_teacher<<" ";
        solution.m_schedule[c][ts_subset[j]] = r[i];
        break;
      }
    }
  }
  //cout<<endl;
}

string toString(unsigned int a)
{
  stringstream ss;
  ss << a;
  return ss.str();
}


void printSolution(sol_format &solution)
{
  cout<< setw(3)<<left<<"";
  for(unsigned int i = 0; i < solution.m_num_ts; i++)
  {
    cout<<setw(4)<<left<<i;
  }
  cout<<endl;
  for(unsigned int i = 0; i < solution.m_num_q; i++)
  {
    cout<< setw(2) << left;
    cout<<i<<" ";
    for(unsigned int j = 0; j < solution.m_num_ts; j++)
    {
      cout<<setw(4)<<left;
      if(solution.m_schedule[i][j] != NULL)
      cout<<toString(solution.m_schedule[i][j]->m_teacher);
      else
      cout<<"X";
      cout<<"";
    }
    cout<<endl;
  }
}

void convertMatrix(vector<int>& row, vector<vector<ll> > &real_matrix, unsigned int n, unsigned int m)
{
  real_matrix.clear();
  real_matrix.resize(n);
  for(unsigned int i = 0; i < n; i++)
  {
    real_matrix[i].resize(m,0);
  }

  for(unsigned int i = 0; i < row.size(); i++)
  {
    real_matrix[i][row[i]] = 1;
  }
  return;
}

sol_format localSearchMT(High_school* &high_school, sol_format solution, unsigned int m)
{
  volatile unsigned int cost, i,c;
  int how_many;
  vector<unsigned int> subset_ts;
  vector<Requirement*> subset_r;
  vector<int> row_assi_matrix;
  vector<vector<ll> > cost_matrix, assignment_matrix;
  sol_format solution_holder;
  //Hungarian hungarian;
  //cout<<"NUEVO"<<endl;
  do
  {
    cost = solution.m_score;
    cout<<"mi cost: "<<cost<<endl;
    i = m*high_school->m_classes;
    while(i > 0)
    {
      //cout<<i<<endl;
      //cout<<"empieza subbucle"<<endl;
      subset_ts.clear(); subset_r.clear();
      c = rand() % high_school->m_classes;
      how_many = rand() % solution.m_num_ts;
      if(how_many == 0) how_many = 1;
      chooseRandomSubset(subset_ts, solution.m_num_ts, how_many);
      getRequirements(solution, subset_ts,subset_r,c);

      /*cout<<"reqs."<<endl;
      for(unsigned int p = 0; p < subset_r.size(); p++)
      {
      cout<<subset_r[p]->m_teacher<<" ";
    }
    cout<<endl;*/

    //cout<<"obtendre cost_matrix"<<endl;
    cost_matrix = calculateCostMatrix(high_school,solution,subset_ts,subset_r,c);

    //cout<<"obtengo cost_matrix de tam: "<<cost_matrix.size()<<endl;
    //solve MCAP
    HungarianAlgorithm hungarian;
    hungarian.Solve(cost_matrix,row_assi_matrix);
    convertMatrix(row_assi_matrix,assignment_matrix,cost_matrix.size(), cost_matrix.size());



    //cout<<"solved"<<endl;
    //update

    /*cout<<"reqs2"<<endl;
    for(unsigned int p = 0; p < subset_r.size(); p++)
    {
    cout<<subset_r[p]->m_teacher<<" ";
  }
  cout<<endl;

  for(unsigned int p = 0; p < assignment_matrix.size(); p++)
  {
  for(unsigned int pp = 0; pp < assignment_matrix[p].size(); pp++)
  {
  cout<<assignment_matrix[p][pp]<<" ";
}
cout<<endl;
}*/

updateSolution(high_school,solution,assignment_matrix,subset_ts,subset_r,c);
//printSolution(solution);

//cout<<"updated"<<endl;
fitness(high_school,&solution);
i--;
}
//cout<<"mi costo: "<<solution.m_score<<" comparado con mejor costo: "<<cost<<endl;
//if((unsigned int)solution.m_score >= (unsigned int)cost) break;
}while(solution.m_score < cost);
return solution;
}

sol_format localSearchTQ(High_school* &high_school, sol_format solution)
{
  vector<vector<unsigned int> > graph;
  vector<vector<unsigned int> > components;

  sol_format new_solution;
  unsigned int best;
  string dummy;
  do{

    best = solution.m_score;
    for(unsigned int i = 0; i < solution.m_num_ts; i++)
    {
      for(unsigned int j = 0; j < solution.m_num_ts; j++)
      {
        /*cin>>dummy;
        cout<<i<<" "<<j<<endl;*/
        if(i != j)
        {
          /*cout<<"the current solution: "<<endl;
          printSolution(solution);
          cout<<endl;*/
          graph.clear();

          TQMove(&solution, graph, i, j);

          /*cout<<"Its graph: "<<endl;
          for(unsigned int ii = 0; ii < graph.size(); ii++)
          {
          cout<<ii<<": ";
          for(unsigned int jj = 0; jj < graph[ii].size(); jj++)
          {
          cout<<graph[ii][jj]<<" ";
        }
        cout<<endl;
      }*/

      components.clear();
      getComponents(graph,components);

      /*cout<<"Its components: "<<endl;
      for(unsigned int ii = 0; ii < components.size(); ii++)
      {
      cout<<ii<<": ";
      for(unsigned int jj = 0; jj < components[ii].size(); jj++)
      {
      cout<<components[ii][jj]<<" ";
    }
    cout<<endl;
  }*/

  for(unsigned int k = 0; k < components.size(); k++)
  {
    //cin>>dummy;
    new_solution = applySwap(solution,components[k], i, j);
    fitness(high_school,&new_solution);
    if(new_solution.m_score <= solution.m_score)
    {
      /* cout<<"solution changed in: "<<k<<endl;
      cout<<"Old - New"<<endl;
      printSolution(solution);
      cout<<solution.m_score<<endl<<endl;*/
      solution = new_solution;
      /*printSolution(new_solution);
      cout<<solution.m_score<<endl<<endl;*/
    }

  }
}

}
}

}while(solution.m_score < best);
return solution;
}




sol_format perturbation(High_school* high_school, sol_format solution)
{
  vector<vector<unsigned int> > graph, components;
  unsigned int ti = rand() % solution.m_num_ts;
  unsigned int tj = rand() % solution.m_num_ts;
  while(ti == tj) tj = rand() % solution.m_num_ts;
  unsigned int k;
  sol_format new_solution;
  //cout<<"cambiando t: "<<ti<<" "<<tj<<endl;
  TQMove(&solution, graph, ti, tj);

  /*cout<<"my graph: "<<endl;
  for(unsigned int i = 0; i < graph.size(); i++)
  {
  cout<<i<<": ";
  for(unsigned int j = 0; j < graph[i].size(); j++)
  {
  cout<<graph[i][j]<<" ";
}
cout<<endl;
}*/
//cout<<"finish TQ"<<endl;
getComponents(graph, components);


//cout<<"components got"<<endl;

/*for(k = 0; k < components.size(); k++)
{
new_solution = applySwap(solution,components[k],ti,tj);
fitness(high_school,&solution);
if(new_solution.m_score <= solution.m_score)
{
solution = new_solution;
}
}*/

k = rand() % components.size();

/*cout<<"mi componente es: "<<endl;
for(unsigned int i = 0; i < components[k].size(); i++)
{
cout<<components[k][i]<<" ";
}
cout<<endl;*/

return new_solution = applySwap(solution,components[k],ti,tj);

}

sol_format VNS_MT_TQ(High_school* high_school, sol_format solution, unsigned int tmax, int kmax)
{
  int k;
  clock_t finale;
  double total_time;
  fitness(high_school,&solution);
  sol_format best_solution = solution;
  while((unsigned int)total_time < tmax)
  {
    k = 1;
    cout<<best_solution.m_score<<endl;
    do
    {
      solution = perturbation(high_school,solution);
      if(k <= kmax - 1)
      solution = localSearchMT(high_school,solution,k);
      else
      //localSearchMT(high_school,solution,k);
      solution = localSearchTQ(high_school,solution);
      fitness(high_school,&solution);
      if(solution.m_score < best_solution.m_score)
      k = 1;
      else
      k++;
      if(solution.m_score <= best_solution.m_score){
        //cout<<"EL MEJOR"<<endl;
        best_solution = solution;
      }
      else{
        solution = best_solution;
      }
      finale = clock();
      total_time = double(finale - start) / CLOCKS_PER_SEC;
    }while(k <= kmax);
  }
  return best_solution;
}

sol_format iteratedLocalSearchTQ(High_school* high_school,sol_format solution, unsigned int stop_condition)
{
  unsigned int not_improved = 0;
  fitness(high_school,&solution);
  sol_format best_solution = solution;
  clock_t finale;
  double total_time;
  while((unsigned int)total_time < stop_condition)
  {
    cout<<best_solution.m_score<<endl;
    //cout<<i<<endl;
    //cout<<"ini perturbation"<<endl;
    solution = perturbation(high_school, solution);
    //cout<<"perturbed"<<endl;
    fitness(high_school, &solution);
    solution = localSearchTQ(high_school,solution);

    if(isBetter(&solution, &best_solution))
    {
      //best_solution = solution;
      not_improved = 0;
    }
    else not_improved++;

    if(solution.m_hc_score+solution.m_sf_score < best_solution.m_hc_score+best_solution.m_sf_score)
    {
      best_solution = solution;
    }
    if(not_improved >= 3)
    {
      solution = best_solution;
      not_improved = 0;
    }
    finale = clock();
    total_time = double(finale - start) / CLOCKS_PER_SEC;
  }
  return best_solution;
}







struct test
{
  int* a;
  vector<int> aa;
};

//

int main()
{


  /*test b;
  b.a = new int(4);
  b.aa.push_back(4);
  cout<<*(b.a)<<" "<<b.aa[0]<<endl;
  test c = b;
  cout<<*(c.a)<<" "<<c.aa[0]<<endl;
  *(b.a) = 40;
  b.aa[0] = 40;
  cout<<*(c.a)<<" "<<c.aa[0]<<endl;*/

  string names[34] = {"CL-CEASD-2008-V-A", "CL-CEASD-2008-V-B",
  "CL-CECL-2011-M-A", "CL-CECL-2011-M-B",
  "CL-CECL-2011-N-A", "CL-CECL-2011-V-A",
  "CM-CECM-2011-M", "CM-CECM-2011-N",
  "CM-CECM-2011-V", "CM-CEDB-2010-N",
  "CM-CEUP-2008-V", "CM-CEUP-2011-M",
  "CM-CEUP-2011-N", "CM-CEUP-2011-V",
  "FA-EEF-2011-M", "JNS-CEDPII-2011-M",
  "JNS-CEDPII-2011-V", "JNS-CEJXXIII-2011-M",
  "JNS-CEJXXIII-2011-N", "JNS-CEJXXIII-2011-V",
  "MGA-CEDC-2011-M", "MGA-CEDC-2011-V",
  "MGA-CEGV-2011-M", "MGA-CEGV-2011-V",
  "MGA-CEJXXIII-2010-V", "MGA-CEVB-2011-M",
  "MGA-CEVB-2011-V", "NE-CESVP-2011-M-A",
  "NE-CESVP-2011-M-B","NE-CESVP-2011-M-C",
  "NE-CESVP-2011-M-D","NE-CESVP-2011-V-A",
  "NE-CESVP-2011-V-B","NE-CESVP-2011-V-C"};

  srand(time(NULL));
  string path;
  unsigned int seconds = 60 * 10; //10 min
  ofstream file("results_MT_4.txt", fstream::app);


  for(unsigned int i = 0; i < 34; i++)
  {
    High_school high_school;
    path = "instances/" + names[i] + ".xml";
    readXML(high_school, path);
    //cout<<high_school.m_periods<<endl;
    initializeTeacher(high_school.m_teachers, high_school.m_periods * high_school.m_days);
    initializeAssignment(high_school.requirements.size(),high_school.m_days);
    initializeTeacherWD(high_school.m_teachers, high_school.m_days);
    sol_format solution;
    //cin>>seconds;

    start = clock();

    solution = generateSolution(&high_school, high_school.requirements);
    //cout<<"SOLUCION INICIAL"<<endl;
    printSolution(solution);
    cout<<solution.m_score<<endl;

    //fitness(&high_school,solution);

    //cout<<solution.m_hc_score<<" + "<<solution.m_sf_score<<" = "<<solution.m_hc_score+solution.m_sf_score<<endl;
    solution = iteratedLocalSearchTQ(&high_school, solution, seconds);
    //solution = VNS_MT_TQ(&high_school,solution,seconds,7);

    printSolution(solution);
    cout<<solution.m_score<<endl;
    //cout<<solution.m_hc_score<<" + "<<solution.m_sf_score<<" = "<<solution.m_hc_score+solution.m_sf_score<<endl;
    file << i+1 << " " << solution.conflicts_teachers << " " << solution.availabilities << " " << solution.max_lessons_day << " ";
    file << solution.minimum_double_lessons << " " << solution.idle_times << " " << solution.teacher_compactness << " " << solution.m_score << '\n';

    delete teacher_timeslots;
    delete assignments;
    delete teacher_wd;
    teacher_timeslots = NULL;
    assignments = NULL;
    teacher_wd = NULL;
  }
  file.close();
  return 0;
}

#endif

int main()
{


  /*test b;
  b.a = new int(4);
  b.aa.push_back(4);
  cout<<*(b.a)<<" "<<b.aa[0]<<endl;
  test c = b;
  cout<<*(c.a)<<" "<<c.aa[0]<<endl;
  *(b.a) = 40;
  b.aa[0] = 40;
  cout<<*(c.a)<<" "<<c.aa[0]<<endl;*/

  string names[34] = {"CL-CEASD-2008-V-A", "CL-CEASD-2008-V-B",
  "CL-CECL-2011-M-A", "CL-CECL-2011-M-B",
  "CL-CECL-2011-N-A", "CL-CECL-2011-V-A",
  "CM-CECM-2011-M", "CM-CECM-2011-N",
  "CM-CECM-2011-V", "CM-CEDB-2010-N",
  "CM-CEUP-2008-V", "CM-CEUP-2011-M",
  "CM-CEUP-2011-N", "CM-CEUP-2011-V",
  "FA-EEF-2011-M", "JNS-CEDPII-2011-M",
  "JNS-CEDPII-2011-V", "JNS-CEJXXIII-2011-M",
  "JNS-CEJXXIII-2011-N", "JNS-CEJXXIII-2011-V",
  "MGA-CEDC-2011-M", "MGA-CEDC-2011-V",
  "MGA-CEGV-2011-M", "MGA-CEGV-2011-V",
  "MGA-CEJXXIII-2010-V", "MGA-CEVB-2011-M",
  "MGA-CEVB-2011-V", "NE-CESVP-2011-M-A",
  "NE-CESVP-2011-M-B","NE-CESVP-2011-M-C",
  "NE-CESVP-2011-M-D","NE-CESVP-2011-V-A",
  "NE-CESVP-2011-V-B","NE-CESVP-2011-V-C"};

  //srand(time(NULL));
  string path;
  unsigned int seconds = 60 * 10; //10 min
  ofstream file("results_MT_4.txt", fstream::app);


  Solution sol, bsol;
  int res, bres = numeric_limits<decltype(bres)>::max();

  cout<<"entra"<<endl;
  for(unsigned int i = 0; i < 34; i++)
  {
    cout<<names[i]<<endl;
    path = "instances/" + names[i] + ".xml";
    readXML(path);

    cout<<"done reading."<<endl;
    iteratedLocalSearchTQ(sol, 20);

    cout<<"solution generated"<<endl;
    for(auto & s : sol.schd)
    {
      int d = -1;
      if (s<reqs.end()) d = s - reqs.begin();
      cout<< d <<" ";
    }

    cout << sol.total << endl;
    /*if ( res < bres)
    {
    bres = res;
    bsol = sol;
  }*/

}

file.close();
return 0;
}
