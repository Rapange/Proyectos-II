#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <ctime>
#include <string>
#include <iomanip>
#include "hungarian.hpp"

using namespace std;
typedef unsigned int ll;


struct Solution;

unsigned int **teacher_timeslots, **assignments, **teacher_wd;
clock_t start;
unordered_map<string,int> q_name_idx, r_name_idx, t_name_idx, c_name_idx;

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


struct Solution{
    vector<vector<Requirement*> > m_schedule;
    vector<vector<unsigned int> > m_available_ts;
    unsigned int m_num_q, m_num_ts;
    volatile unsigned int m_hc_score, m_sf_score, m_score;
    volatile unsigned int max_lessons_day, conflicts_teachers, availabilities;
    volatile unsigned int minimum_double_lessons, idle_times, teacher_compactness;

    Solution(unsigned int num_q, unsigned int num_ts)
    {
        m_num_q = num_q;
        m_num_ts = num_ts;
        m_hc_score = 0;
        m_sf_score = 0;
        m_score = 0;
        max_lessons_day = conflicts_teachers = availabilities = 0;
        minimum_double_lessons = idle_times = teacher_compactness = 0;
        m_schedule.resize(m_num_q);

        vector<unsigned int> available;
        for(unsigned int i = 0; i < m_num_ts; i++)
        {
            //cout<<"llena: "<<i<<endl;
            available.push_back(i);
        }

        for(unsigned int i = 0; i < m_num_q; i++)
        {
            m_schedule[i].resize(m_num_ts);
            for(unsigned int j = 0; j < m_num_ts; j++)
            {
                m_schedule[i][j] = NULL;
            }
            //cout<<"push: "<<i<<endl;
            m_available_ts.push_back(available);
        }
    }
    Solution()
    {

    }
};

typedef Solution sol_format;

void initializeTeacher(unsigned int num_t, unsigned int num_ts)
{
    teacher_timeslots = new unsigned int*[num_t];
    for(unsigned int i = 0; i < num_t; i++)
    {
        teacher_timeslots[i] = new unsigned int[num_ts];
    }

    myFill(teacher_timeslots,num_t,num_ts,0);


}

void initializeAssignment(unsigned int num_r, unsigned int num_days)
{
    assignments = new unsigned int*[num_r];
    for(unsigned int i = 0; i < num_r; i++)
    {
        assignments[i] = new unsigned int[num_days];
    }

    myFill(assignments, num_r, num_days, 0);
}

void initializeTeacherWD(unsigned int num_t, unsigned int num_days)
{
    teacher_wd = new unsigned int*[num_t];
    for(unsigned int i = 0; i < num_t; i++)
    {
        teacher_wd[i] = new unsigned int[num_days];
    }

    myFill(teacher_wd, num_t, num_days, 0);
}


void fillTeachers(High_school* &high_school, sol_format* &solution)
{
    unsigned int teacher_idx = 0;

    myFill(teacher_timeslots,high_school->m_teachers, solution->m_num_ts,0);


    for(unsigned int i = 0; i < solution->m_num_q; i++)
    {
        for(unsigned int j = 0; j < solution->m_num_ts; j++)
        {
            //cout<<j<<endl;
            if(solution->m_schedule[i][j] != NULL)
            {
                //cout<<t_name_idx[solution->m_schedule[j][i]->m_teacher]<<" "<<r_name_idx[solution->m_schedule_rooms[j][i]->m_name]<<endl;
                //cout<<"teacher id en pos: "<<i<<" "<<j<<" ";
                teacher_idx = solution->m_schedule[i][j]->m_teacher;
                //cout<<teacher_idx<<endl;
                teacher_timeslots[teacher_idx][j]++;
            }

        }
    }
}

void fillAssignments(High_school* &high_school, sol_format* &solution)
{
    Requirement* requirement;
    myFill(assignments,high_school->requirements.size(), high_school->m_days, 0);
    for(unsigned int i = 0; i < solution->m_num_q; i++)
    {
        for(unsigned int j = 0; j < solution->m_num_ts; j++)
        {
            requirement = solution->m_schedule[i][j];
            if(requirement)
                assignments[requirement->m_id][j / high_school->m_periods]++;
        }
    }

    return;
}

void fillTeacherWD(High_school* &high_school, sol_format* &solution)
{
    Requirement* requirement;
    myFill(teacher_wd, high_school->m_teachers, high_school->m_days,0);
    for(unsigned int i = 0; i < solution->m_num_q; i++)
    {
        for(unsigned int j = 0; j < solution->m_num_ts; j++)
        {
            requirement = solution->m_schedule[i][j];
            if(requirement)
                teacher_wd[requirement->m_teacher][j / high_school->m_periods] = 1;
        }
    }
    return;
}

unsigned int teacherCompactness(High_school* &high_school, sol_format* solution)
{
    unsigned int sum = 0;
    fillTeacherWD(high_school,solution);

    for(unsigned int i = 0; i < high_school->m_teachers; i++)
    {
        for(unsigned int j = 0; j < high_school->m_days; j++)
        {
            //cout<<"teacher "<<i<<" day "<<j<<" = "<<teacher_wd[i][j]<<endl;
            sum += teacher_wd[i][j];
        }
    }
    return sum;
}

unsigned int conflicts(High_school* &high_school, sol_format* solution)
{
    fillTeachers(high_school,solution);

    unsigned int sum = 0;
    for(unsigned int i = 0; i < high_school->m_teachers; i++)
    {
        //cout<<"teacher "<<i<<": ";
        for(unsigned int j = 0; j < solution->m_num_ts; j++)
        {
            //cout<<teacher_timeslots[i][j]<<" ";
            if(teacher_timeslots[i][j] > 1)
            {
                sum += teacher_timeslots[i][j] - 1;
            }

        }
        //cout<<endl;
    }
    return sum;
}

unsigned int availabilities(High_school* &high_school, sol_format* &solution)
{
    unsigned int sum = 0, unavailable_timeslot, timeslot;
    Teacher_unavailability *teacher_unavailability;
    for(unsigned int i = 0; i < high_school->teacher_unavailabilities.size(); i++)
    {
        teacher_unavailability = high_school->teacher_unavailabilities[i];
        for(unsigned int j = 0; j < solution->m_num_q; j++)
        {
            timeslot = teacher_unavailability->m_day * high_school->m_periods + teacher_unavailability->m_period;
            if(solution->m_schedule[j][timeslot] &&
               solution->m_schedule[j][timeslot]->m_teacher == teacher_unavailability->m_teacher)
            {
                sum++;
            }
        }

    }
    return sum;
}

unsigned int maxLessonsDay(High_school* &high_school, sol_format* &solution)
{
    unsigned int sum = 0, max_lessons_day = 0, assigned_lessons_day = 0;

    fillAssignments(high_school, solution);
    for(unsigned int i = 0; i < high_school->requirements.size(); i++)
    {
        max_lessons_day = high_school->requirements[i]->m_max_per_day;
        for(unsigned int j = 0; j < high_school->m_days; j++)
        {
            assigned_lessons_day = assignments[i][j];

            if(assigned_lessons_day > max_lessons_day)
            {
                //cout<<"Para: "<<i<<" dia: "<<j<<" max lessons: "<<max_lessons_day<<" y assigned: "<<assigned_lessons_day<<endl;
                sum += (assigned_lessons_day - max_lessons_day);
            }
        }
    }
    return sum;
}

inline unsigned int getDoubleLessons(High_school* &high_school, sol_format* &solution, Requirement* requirement)
{
    unsigned int double_lessons = 0, q = requirement->m_class;
    unsigned int ts_per_day = high_school->m_periods;
    for(unsigned int i = 0; i < solution->m_num_ts - 1; i++)
    {
        if(solution->m_schedule[q][i] == requirement)
           if(solution->m_schedule[q][i+1] == requirement && i / ts_per_day == (i+1) / ts_per_day)
                double_lessons++;
    }

    return double_lessons;
}

unsigned int minDoubleLessons(High_school* &high_school, sol_format* &solution)
{
    Requirement* requirement;
    unsigned int double_lessons = 0, sum = 0;
    for(unsigned int i = 0; i < high_school->requirements.size(); i++)
    {
        requirement = high_school->requirements[i];
        if(requirement)
        {
            double_lessons = getDoubleLessons(high_school,solution,requirement);
            //cout<<"double lessons: "<<double_lessons<<" para req: "<<i<<" y min double lessons: "<<requirement->m_double_lessons<<endl;
            if(double_lessons < requirement->m_double_lessons)
            {
                sum += (requirement->m_double_lessons - double_lessons);
            }
        }

    }
    return sum;
}

inline unsigned int getIdleTeacher(High_school* &high_school, sol_format* &solution, unsigned int day, unsigned int teacher_id)
{
    unsigned int num_teacher = 0, current_idle_times = 0, holder_idle_times = 0;
    bool teacher_in_day = false;
    for(unsigned int i = day * high_school->m_periods; i < (day + 1) * high_school->m_periods; i++)
    {
        teacher_in_day = false;
        for(unsigned int j = 0; j < solution->m_num_q; j++)
        {
            if(solution->m_schedule[j][i] && solution->m_schedule[j][i]->m_teacher == teacher_id)
            {
                teacher_in_day = true;
                break;

            }

        }
        if(teacher_in_day)
        {
            current_idle_times += holder_idle_times;
            holder_idle_times = 0;
            num_teacher++;
        }
        else if(num_teacher > 0)
        {
            holder_idle_times++;
        }

    }
    return current_idle_times;
}

unsigned int idleTimes(High_school* &high_school, sol_format* &solution)
{
    unsigned int sum = 0;
    for(unsigned int i = 0; i < high_school->m_teachers; i++)
    {
        for(unsigned int j = 0; j < high_school->m_days; j++)
        {
            //cout<<"para teacher "<<i<<" y dia "<<j<<" = "<<getIdleTeacher(high_school,solution,j,i)<<endl;
            sum += getIdleTeacher(high_school,solution,j,i);
        }
    }
    return sum;
}

int getHardConstraints(High_school* &high_school, sol_format* solution)
{
    //cout<<"hard constraints"<<endl;
    unsigned int mld = maxLessonsDay(high_school, solution);
    //cout<<"max lessons day: "<<mld<<endl;
    unsigned int c = conflicts(high_school, solution);
    //cout<<"conflicts teachers: "<<c<<endl;
    unsigned int a = availabilities(high_school,solution);
    //cout<<"availabilities: "<<a<<endl<<endl;

    mld *= 10000; c *= 100000; a *= 100000;
    solution->max_lessons_day = mld;
    solution->conflicts_teachers = c;
    solution->availabilities = a;
    return c+ a+ mld;

}

int getSoftConstraints(High_school* &high_school, sol_format* solution)
{
    //cout<<"soft constraints: "<<endl;
    unsigned int tc = teacherCompactness(high_school,solution);
    //cout<<"teacher compactness: "<<tc<<endl;
    unsigned int mdl = minDoubleLessons(high_school,solution);
    //cout<<"minimum double lessons: "<<mdl<<endl;
    unsigned int it = idleTimes(high_school,solution);
    //cout<<"idle times: "<<it<<endl<<endl;

    tc *= 9; mdl *= 1; it *= 3;

    solution->minimum_double_lessons = mdl;
    solution->teacher_compactness = tc;
    solution->idle_times = it;
    return mdl+it+tc;
}

inline void TQMove(sol_format* solution, vector<vector<unsigned int> > &graph, int ti, int tj)
{
    vector<unsigned int> node;
    bool i_ti_is_NULL, i_tj_is_NULL, j_ti_is_NULL, j_tj_is_NULL;
    //graph.clear();
    for(unsigned int i = 0; i < solution->m_num_q; i++)
    {
        node.clear();
        if(solution->m_schedule[i][ti] == NULL) i_ti_is_NULL = true;
        else i_ti_is_NULL = false;

        if(solution->m_schedule[i][tj] == NULL) i_tj_is_NULL = true;
        else i_tj_is_NULL = false;

        if(i_ti_is_NULL && i_tj_is_NULL);
        else
            for(unsigned int j = i+1; j < solution->m_num_q; j++)
            {
                if(solution->m_schedule[j][ti] == NULL) j_ti_is_NULL = true;
                else j_ti_is_NULL = false;

                if(solution->m_schedule[j][tj] == NULL) j_tj_is_NULL = true;
                else j_tj_is_NULL = false;

                if(i_ti_is_NULL && !i_tj_is_NULL && !j_ti_is_NULL)
                {
                    if(solution->m_schedule[j][ti]->m_teacher == solution->m_schedule[i][tj]->m_teacher)
                    {
                        node.push_back(j);
                    }

                }

                else if(!i_ti_is_NULL && i_tj_is_NULL && !j_tj_is_NULL)
                {
                    if(solution->m_schedule[j][tj]->m_teacher == solution->m_schedule[i][ti]->m_teacher)
                    {
                        node.push_back(j);
                    }
                }

                else if(!i_ti_is_NULL && !i_tj_is_NULL)
                {
                    if(!j_ti_is_NULL && !j_tj_is_NULL)
                    {
                        if(solution->m_schedule[j][ti]->m_teacher == solution->m_schedule[i][tj]->m_teacher ||
                       solution->m_schedule[j][tj]->m_teacher == solution->m_schedule[i][ti]->m_teacher)
                        {
                            node.push_back(j);
                        }
                    }
                    else if(!j_ti_is_NULL && j_tj_is_NULL)
                    {
                        if(solution->m_schedule[j][ti]->m_teacher == solution->m_schedule[i][tj]->m_teacher)
                        {
                            node.push_back(j);
                        }
                    }
                    else if(j_ti_is_NULL && !j_tj_is_NULL)
                    {
                        if(solution->m_schedule[j][tj]->m_teacher == solution->m_schedule[i][ti]->m_teacher)
                        {
                            node.push_back(j);
                        }
                    }
                }



            }
        graph.push_back(node);
    }
    return;
}

void DFS(vector<vector<unsigned int> > &graph, unsigned int start, vector<unsigned int> &component, vector<bool> &visited)
{
    vector<int> myStack;
    //component.clear();
    myStack.push_back(start);
    while(!myStack.empty())
    {
        start = myStack.back();
        visited[start] = 1;
        component.push_back(start);
        myStack.pop_back();
        for(unsigned int i = 0; i < graph[start].size(); i++)
        {
            if(!visited[graph[start][i]])
                myStack.push_back(graph[start][i]);
        }
    }
    return;
}

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


void fitness(High_school* &high_school, sol_format* solution)
{
    solution->m_hc_score = getHardConstraints(high_school,solution);
    solution->m_sf_score = getSoftConstraints(high_school,solution);
    solution->m_score = solution->m_hc_score + solution->m_sf_score;
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
        //cout<<"mi cost: "<<cost<<endl;
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

sol_format generateSolution(High_school* high_school, vector<Requirement*> &r)
{
    Requirement* requirement;
    unsigned int num_q = high_school->m_classes, num_ts = (high_school->m_days ) * (high_school->m_periods );
    sol_format solution(num_q, num_ts);
    //solution = new sol_format(num_q, num_ts);
    int lectures = 0, q;
    int random_ts, random_room;
    //cout<<num_q<<" "<<num_ts<<endl;
    for(unsigned int i = 0; i < r.size(); i++)
    {
        requirement = r[i];

        q = requirement->m_class;
        lectures = requirement->m_lessons;
        //cout<<"Clase: "<<q<<" "<<lectures<<endl;
        while(lectures > 0)
        {
            //cout<<"eligiendo random de: "<<solution.m_available_ts[q].size()<<endl;
            random_ts = rand() % solution.m_available_ts[q].size();

            //cout<<"random elegido: "<<random_ts<<endl;
            solution.m_schedule[q][solution.m_available_ts[q][random_ts]] = requirement;


            solution.m_available_ts[q].erase(solution.m_available_ts[q].begin()+random_ts);

            //cout<<"end"<<endl;
            lectures--;
        }

    }
    fitness(high_school,&solution);
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
        //cout<<best_solution.m_score<<endl;
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
        //cout<<best_solution.m_score<<endl;
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



bool isNum(char a)
{
    if(a == '0' || a == '1' || a == '2' || a == '3' || a == '4' || a == '5' || a == '6' || a == '7' || a == '8' || a == '9') return true;
    return false;
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






void readXML(High_school &high_school, string filename)
{
    vector<unsigned int> values;
    string word;
    ifstream file(filename);
    unsigned int id = 0;
    //High_school high_school;

    for(unsigned int i = 0; i < 4; i++)
    {
        file >> word;
        //cout<<word<<endl;
    }
    for(unsigned int i = 0; i < 4; i++)
    {
        //values.clear();
        getline(file,word);
        //cout<<word<<endl;
        readXMLLine(word,values);
        /*for(unsigned int j = 0; j < values.size(); j++)
        {
            cout<<values[j]<<endl;
        }*/
    }

    high_school.m_classes = values[1] + 1;
    high_school.m_teachers = values[3] + 1;
    high_school.m_days = values[5] + 1;
    high_school.m_periods = values[7] + 1;

    for(unsigned int i = 0; i < 2; i++) file >> word;

    Requirement* requirement;
    while(file >> word && word != "</requirements>")
    {
        requirement = new Requirement;
        values.clear();
        getline(file,word);
        //cout<<word<<endl;
        readXMLLine(word,values);
        /*for(unsigned int j = 0; j < values.size(); j++)
        {
            cout<<values[j]<<endl;
        }*/
        requirement->m_class = values[0];
        requirement->m_teacher = values[1];
        requirement->m_lessons = values[2];
        requirement->m_max_per_day = values[3];
        requirement->m_double_lessons = values[4];
        requirement->m_id = id;
        id++;
        high_school.requirements.push_back(requirement);
    }

    file >> word;

    Teacher_unavailability* teacher_unavailability;
    while(file >> word && word != "</teacherunavailabilities>")
    {
        teacher_unavailability = new Teacher_unavailability;
        values.clear();
        getline(file,word);
        //cout<<word<<endl;
        readXMLLine(word,values);
        /*for(unsigned int j = 0; j < values.size(); j++)
        {
            cout<<values[j]<<endl;
        }*/

        teacher_unavailability->m_teacher = values[0];
        teacher_unavailability->m_day = values[1];
        teacher_unavailability->m_period = values[2];

        high_school.teacher_unavailabilities.push_back(teacher_unavailability);
    }
    file.close();
}

int strToInt(string num)
{
    int my_num;
    stringstream ss(num);
    ss >> my_num;
    return my_num;
}

//

int main()
{


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
    string path, index;
    unsigned int seconds = 5; //10 min

    ofstream file("results.txt", fstream::app);

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
        //printSolution(solution);

        //fitness(&high_school,solution);

        //cout<<solution.m_hc_score<<" + "<<solution.m_sf_score<<" = "<<solution.m_hc_score+solution.m_sf_score<<endl;
        solution = iteratedLocalSearchTQ(&high_school, solution, seconds);
        //solution = VNS_MT_TQ(&high_school,solution,seconds,7);

        //printSolution(solution);
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
