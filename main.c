#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
#include <math.h>
#include <immintrin.h>
#include "dSFMT.h"
#define SQR(x) ((x)*(x))
#define DIST_MAX 999
#define PATH_SIZE 32
#define TRACE 10
dsfmt_t dsfmt;
typedef struct branch{
	int length; //branching length
	int size;
	int *taken;
	int *all;
}branch;
typedef struct node{
	int flag;
	int pass;
	int orientation;
	int nbond; //number of bonds
	double *angle;
	void **c; //pointer to a cycle
	struct node **next;
}node;
typedef struct edge{
	int num;
	int flag; //xor for a particular edge
	int i,j;	
	unsigned int xor; //xor for a particular edge
	node *pi,*pj; //nodes pi and pj
	struct edge *inverse; //if an edge in opposite direction exists
}edge;
typedef struct stack_edge{
	int length;
	unsigned int *trace;
	edge **e;
}stack_edge;
typedef struct path{
	int length; //length of the path
	int nbranch; //number of branching
	int i,j;	
	edge *e;
	struct path **pleft,**pright;
}path;
typedef struct cycle_path{
	int length;
	path *first;
	path *second;
	struct cycle_path *left,*right;
}cycle_path;
typedef struct cycle_path_set{
	int size;
	unsigned int size_allocated;
	cycle_path **c;
	cycle_path *head;
}cycle_path_set;
typedef struct cycle{
	int flag;
	int length;
	int *na; //angle along the cycle
	unsigned int *trace;
	node **p;
	edge **e;
}cycle;
typedef struct matrix{
	int size;
	node *p;
	branch *b;
	stack_edge *s;
	cycle_path_set *cset;
	int ncycles;
	cycle *c;
	int nedges,nxor;
	edge *edges;
	int **graph; //graph, adjency matrix
	int **dist; //distance matrix
	int **order;
	path **first;
	path **second;
}matrix;
//Aritmetics
double length2(__m128d x){
	x=_mm_dp_pd(x,x,0xFF);
	return _mm_cvtsd_f64(x);
}
double length(__m128d x){
	x=_mm_dp_pd(x,x,0xFF);
	return sqrt(_mm_cvtsd_f64(x));
}
double dot(__m128d a,__m128d b){
	__m128d c=_mm_dp_pd(a,b,0x31);
	return _mm_cvtsd_f64(c);
}
__m128d normalize(__m128d a){
	double n=length(a);
	return a/n;
}
__m128d rnd11(){
	__m128d b={dsfmt_genrand_open_open(&dsfmt),dsfmt_genrand_open_open(&dsfmt)};
	return b;
}
inline __m128d sincosa(double a){
	__m128d b={sin(a),cos(a)};
	return b;
}
inline __m128d rot22(__m128d a,__m128d b01){
	__m128d b10={-b01[1],b01[0]};
	return _mm_set1_pd(a[1])*b01-_mm_set1_pd(a[0])*b10;
}
inline __m128d rot2w(__m128d a,double w){
	__m128d b={-sin(w),cos(w)};
	return normalize(rot22(a,b));
}
//
//Memory allocations
int init_path(path *p,int size){
	int i;
	p->pleft=(path**)alloc(sizeof(path*)*size);
	p->pright=(path**)alloc(sizeof(path*)*size);
	//p->pleft=(path**)alloc(sizeof(path*)*64);
	//p->pright=(path**)alloc(sizeof(path*)*64);
	p->e=NULL;
	p->length=0;
	p->nbranch=0;
	for(i=0;i<size;i++){
	//for(i=0;i<64;i++){
		p->pleft[i]=NULL;
		p->pright[i]=NULL;
	}
	return 0;
}
int count_edges(matrix *m){
	int i;
	for(i=0,m->nedges=0;i<SQR(m->size);i++){
		if(*(*m->graph+i))m->nedges++;
	}
	return m->nedges;
}
int init_edges(matrix *m){
	int i,j,k;
	unsigned x;
	edge *e;
	path *p,*q;
	k=count_edges(m);
	m->edges=(edge*)alloc(sizeof(edge)*k);
	k=0,x=0;
	for(i=0;i<m->size;i++){
		for(j=0;j<m->size;j++){
			p=&m->first[i][j];
			if(m->graph[i][j]){
				e=m->edges+k;
				e->i=i;
				e->j=j;
				e->num=k++;
				e->inverse=NULL;
				e->xor=0;

				p->e=e;
				p->nbranch=0;
				p->length=0;
			}
		}
	}
	for(i=0;i<m->size;i++){
		for(j=0;j<m->size;j++){
			p=&m->first[i][j];
			if(p->e){
				q=&m->first[j][i];
				p->e->inverse=q->e;
				if(j>i){
					p->e->xor=x;
					p->e->inverse->xor=x;
					x++;
				}
			}
		}
	}
	m->nxor=x;
	return 0;
}
branch *init_branch(int size){
	int i;
	branch *b=(branch*)alloc(sizeof(branch));
	b->taken=(int*)alloc(sizeof(int)*size);
	b->all=(int*)alloc(sizeof(int)*size);
	b->length=0;
	b->size=size;
	for(i=0;i<size;i++){
		*(b->taken+i)=0;
		*(b->all+i)=0;
	}
	return b;
}
stack_edge *init_stack_edge(int size){
	int i;
	stack_edge *s=(stack_edge*)alloc(sizeof(stack_edge)*size);
	for(i=0;i<size;i++){
		(s+i)->length=0;
		(s+i)->trace=(unsigned int*)alloc(sizeof(unsigned int)*TRACE);
		(s+i)->e=(edge**)alloc(sizeof(edge*)*size);
	}
	return s;
}
cycle *init_cycle(int size){
	int i;
	int size2=size*size;
	cycle *c=(cycle*)alloc(sizeof(cycle)*size2);
	for(i=0;i<size2;i++){
		(c+i)->trace=(unsigned int*)alloc(sizeof(unsigned int)*TRACE);
		(c+i)->e=(edge**)alloc(sizeof(edge*)*size);
	}
	return c;
}
cycle_path_set *init_cycle_path_set(int size){
	int i,size2=size*size;
	cycle_path_set *cset=(cycle_path_set*)alloc(sizeof(cycle_path_set));
	cset->c=(cycle_path**)alloc(sizeof(cycle_path*)*size2);
	*cset->c=(cycle_path*)alloc(sizeof(cycle_path)*size2);
	cset->size_allocated=size2;
	cset->size=0;
	cset->head=NULL;
	for(i=0;i<size2;i++){
		*(cset->c+i)=*cset->c+i;
		(*(cset->c+i))->length=0;
	}
	return cset;
}

matrix *init_matrix(int size){
	int i;
	matrix *m=(matrix*)alloc(sizeof(matrix));
	m->size=size;
	m->p=(node*)alloc(sizeof(node)*size);
	m->graph=(int**)alloc(sizeof(int*)*size);
	m->dist=(int**)alloc(sizeof(int*)*size);
	m->order=(int**)alloc(sizeof(int*)*size);
	m->first=(path**)alloc(sizeof(path*)*size);
	m->second=(path**)alloc(sizeof(path*)*size);
	//single alloc
	*m->graph=(int*)alloc(sizeof(int)*size*size);
	*m->dist=(int*)alloc(sizeof(int)*size*size);
	*m->order=(int*)alloc(sizeof(int)*size*size);

	*m->first=(path*)alloc(sizeof(path)*size*size);
	*m->second=(path*)alloc(sizeof(path)*size*size);
	for(i=0;i<size;i++){
		*(m->graph+i)=*m->graph+i*size;
		*(m->dist+i)=*m->dist+i*size;
		*(m->order+i)=*m->order+i*size;

		*(m->first+i)=*m->first+i*size;
		*(m->second+i)=*m->second+i*size;
	}
	for(i=0;i<size*size;i++){
		*(*m->graph+i)=0;
		*(*m->dist+i)=0;
		*(*m->order+i)=0;

		//init_path(*m->first+i,size);
		//init_path(*m->second+i,size);
		init_path(*m->first+i,PATH_SIZE);
		init_path(*m->second+i,PATH_SIZE);
	}
	m->c=init_cycle(size);
	m->b=init_branch(size);
	m->s=init_stack_edge(size);
	m->cset=init_cycle_path_set(size);
	return m;
}
int graph2dist(matrix *m){
	unsigned int i,j;
	unsigned int n=m->size;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			m->dist[i][j]=m->graph[i][j];
			(&m->first[i][j])->length=m->graph[i][j];
			(&m->second[i][j])->length=0;
			(&m->second[i][j])->nbranch=0;
			if(i!=j){
				if(!m->dist[i][j]){
					m->dist[i][j]=DIST_MAX;
					(&m->first[i][j])->length=DIST_MAX;
				}
			}
			else{
				m->dist[i][j]=0;
				(&m->first[i][j])->length=0;
			}
		}
	}
	return 0;
}
matrix *read_matrix(char *name){
	FILE *f;
	int c __attribute__ ((unused));
	int i;
	int size;
	f=open_file(name,"r");
	c=fscanf(f,"%d",&size);
	matrix *m=init_matrix(size);
	m->size=size;
	for(i=0;i<size*size;i++){
		c=fscanf(f,"%u",*m->graph+i);
	}
	//init_edges(m);
	//graph2dist(m);
	return m;
}
//printg matrices
int print_graph(FILE *f,matrix *m){
	int i,j;
	fprintf(f,"%d\n",m->size);
	for(i=0;i<m->size;i++){
		for(j=0;j<m->size;j++){
			fprintf(f,"%d",m->graph[i][j]);
		}
		fputc('\n',f);
	}
	return 0;
}
int print_dist(FILE *f,matrix *m){
	int i,j;
	fprintf(f,"%d\n",m->size);
	for(i=0;i<m->size;i++){
		for(j=0;j<m->size;j++){
			if(m->dist[i][j]==DIST_MAX)fprintf(f,"*");
			else fprintf(f,"%d",m->dist[i][j]);
		}
		fputc('\n',f);
	}
	return 0;
}
//managing paths
int path_change(path *p,path *ik,path *kj){
	p->length=ik->length+kj->length;
	p->nbranch=1;
	*(p->pleft)=ik;
	*(p->pright)=kj;
	return 0;
}
int path_append(path *p,path *ik,path *kj){
	p->length=ik->length+kj->length;
	p->pleft[p->nbranch]=ik;
	p->pright[p->nbranch]=kj;
	p->nbranch++;
	return 0;
}
int path_copy(path *q,path *p){
	int i;
	for(i=0;i<p->nbranch;i++){
		q->pleft[i]=p->pleft[i];
		q->pright[i]=p->pright[i];
	}
	q->length=p->length;
	q->nbranch=p->nbranch;
	return 0;
}
int path_delete(path *p){
	p->length=0;
	p->nbranch=0;
	return 0;
}
int dump_path(path *p){
	int i;
	path *s,*t;
	printf("p->nbranch %d p->length %d\n",p->nbranch,p->length);
	if(p->length){
		for(i=0;i<p->nbranch;i++){
			s=p->pleft[i];
			t=p->pright[i];
			printf("[%2d,%2d][%2d,%2d] {%2d}{%2d}(%d)\n",s->i,s->j,t->i,t->j,s->length,t->length,p->length);
		}
	}
	return 0;
}
int floydwarshall(matrix *m){
	int i,j,k;
	int n=m->size;
	int dikj;
	for(k=0;k<n;k++){
		for(i=0;i<n;i++){
			for(j=0;j<n;j++){
				if(i!=k&&k!=j){ //Excluding loops
					dikj=m->dist[i][k]+m->dist[k][j];
					if(m->dist[i][j]>dikj){
						if(m->dist[i][j]==(dikj+1)){
							path_copy(&m->second[i][j],&m->first[i][j]);
						}
						else{
							path_delete(&m->second[i][j]);
						}
						path_change(&m->first[i][j],&m->first[i][k],&m->first[k][j]);
						m->dist[i][j]=dikj;
					}
					else if(m->dist[i][j]==(dikj-1)){
						path_append(&m->second[i][j],&m->first[i][k],&m->first[k][j]);
					}
					else if(m->dist[i][j]==dikj){
						path_append(&m->first[i][j],&m->first[i][k],&m->first[k][j]);
					}
				}
			}
		}
	}
	return 0;
}
//branch management
int branch_clear(branch *b){
	int k;
	for(k=0,b->length=0;k<b->size;k++){
		b->taken[k]=0;
		b->all[k]=0;
	}
	return 0;
}
int branch_adjust(branch *b){
	int k=b->length,l;
	for(k=b->length-1;k>=0;k--){
		if(b->taken[k]<b->all[k]-1){
			b->taken[k]++;
			for(l=k+1;l<b->length;l++){
				b->taken[l]=0;
			}
			b->length=0;
			return k+1;
		}
	}
	b->length=0;
	return 0;
}
int store_branch(path *p,stack_edge *s,branch *b){
	int k;
	if(p){
		if(!p->e){
			k=b->taken[b->length];
			b->all[b->length]=p->nbranch;
			b->length++;
			store_branch(p->pleft[k],s,b);
			store_branch(p->pright[k],s,b);
		}
		else *(s->e+s->length++)=p->e;
		return 0;
	}
	else return 1;
}
//traces
void trace_null(unsigned int *t){
	unsigned int i;
	for(i=0;i<TRACE;i++){
		*(t+i)=0;
	}
}
void set_trace(unsigned int *t,unsigned int n){
	unsigned int word=n>>0x5;
	unsigned int s=n&0x1f;
	*(t+word)|=1<<s;		
}
int set_trace_check(unsigned int *t,unsigned int n){
	unsigned int word=n>>0x5;
	unsigned int s=n&0x1f;
	if(*(t+word)&1<<s)return 1;
	*(t+word)|=1<<s;		
	return 0;
}
void print_trace(unsigned int *t){
	int i,j;
	unsigned int s;
	for(j=TRACE-1;j>=0;j--){
		s=*(t+j);
		for(i=TRACE*8-1;i>=0;i--){
			putchar('0'+((s>>i)&0x1));
		}
	}
	putchar('\n');
}
int trace_stack(stack_edge *s){
	int i;
	trace_null(s->trace);
	for(i=0;i<s->length;i++){
		set_trace(s->trace,s->e[i]->xor);
	}
	return 0;
}
int trace_cycle(cycle *c){
	int i;
	trace_null(c->trace);
	for(i=0;i<c->length;i++){
		if(set_trace_check(c->trace,c->e[i]->xor))return 1;
	}
	return 0;
}
int cmp_trace(unsigned int *t,unsigned int *s){
	unsigned int i;
	for(i=0;i<TRACE;i++){
		if(*(t+i)^*(s+i))return 1;
	}
	return 0;
}
int cmp_tstacks(stack_edge *s,int k){
	int i;
	trace_stack(s+k);
	for(i=0;i<k;i++){
		if(!cmp_trace((s+i)->trace,(s+k)->trace))return 1;
	}
	return 0;
}
int cmp_cycles(cycle *c,int k){
	int i;
	for(i=0;i<k;i++){
		if(!cmp_trace((c+i)->trace,(c+k)->trace))return 1;
	}
	return 0;
}
//edge stack manipulations
int cmp_stacks(stack_edge *s,stack_edge *t){
	int i;
	for(i=0;i<s->length;i++){
		if(s->e[i]!=t->e[i]){
			return 0;
		}
	}
	return 1;
}
int cmp_kstacks(stack_edge *s,int k){
	int i;
	stack_edge *t=s+k;
	for(i=0;i<k;i++){
		if(cmp_stacks(s+i,t))return 1;
	}
	return 0;
}
int store_stack(path *p,stack_edge *s,branch *b){
	int k=0;
	branch_clear(b);
	s->length=0;
	if(p->length){
		do{
			store_branch(p,s+k,b);
			if(p->length==(s+k)->length){
				if(!cmp_tstacks(s,k)){
					k++;
				}
			}
			(s+k)->length=0;
		}while(branch_adjust(b));
	}
	return k;
}
int print_stack_edge(FILE *f,stack_edge *s,int k){
	int i,j;
	stack_edge *t;
	for(i=0;i<k;i++){
		t=s+i;
		fprintf(f,"%d",(*t->e)->i);		
		for(j=0;j<t->length;j++){
			fprintf(f,"-%d",t->e[j]->j);		
		}
		fputc('\n',f);
	}
	return 0;
}
//cycle path set manipulation
void btree_insert(cycle_path **node,cycle_path *c){
	if(!*node)*node=c;
	else{
		if((*node)->length>c->length){
			btree_insert(&(*node)->left,c);
		}
		else btree_insert(&(*node)->right,c);
	}
}
int btree_inorder(cycle_path_set *cset,cycle_path *c){
	if(!c)return 1;
	else{
		btree_inorder(cset,c->left);
		*(cset->c+cset->size++)=c;
		btree_inorder(cset,c->right);
		return 0;
	}
}
int btree_sort(cycle_path_set *cset){
	cset->size=0;
	btree_inorder(cset,cset->head);
	return 0;
}
int cycle_path_set_append(cycle_path_set *cset,path *first,path *second){
	cycle_path *c;
	c=*(cset->c+cset->size);
	if(first->nbranch<2&&!second->length)return 1;
	else{
		if(!second->length)c->length=2*first->length;
		else c->length=2*first->length+1;
		c->first=first;
		c->second=second;
		if(!cset->size)cset->head=c;
		else btree_insert(&cset->head,c);
		cset->size++;
		return 0;
	}
}
int cycle_path_store(matrix *m){
	int i,j;
	for(i=0;i<m->size;i++){
		for(j=0;j<m->size;j++){
			if(m->dist[i][j]>0){
				cycle_path_set_append(m->cset,&m->first[i][j],&m->second[i][j]);
			}
		}
	}
	btree_sort(m->cset);
	return 0;
}
//finding cycles
int stack2cycle(stack_edge *s,cycle *c){
	int i;
	for(i=0;i<s->length;i++){
		c->e[c->length++]=s->e[i];
	}
	return 0;
}
int stack2cycle_inverse(stack_edge *s,cycle *c){
	int i;
	for(i=0;i<s->length;i++){
		c->e[c->length++]=s->e[s->length-i-1]->inverse;
	}
	return 0;
}
int print_cycle(FILE *f,cycle *c,int k){
	int i,j;
	cycle *t;
	for(i=0;i<k;i++){
		t=c+i;
		fprintf(f,"[%d] %d",t->length,(*t->e)->i);		
		for(j=0;j<t->length;j++){
			fprintf(f,"-%d",t->e[j]->j);		
		}
		fputc('\n',f);
		//print_trace((c+i)->trace);
	}
	return 0;
}
int cycle_odd(cycle *c,int *n,path *first,stack_edge *s,branch *b){
	int i,j,k;
	k=store_stack(first,s,b);
	for(i=0;i<k;i++){
		for(j=i+1;j<k;j++){
			(c+*n)->length=0;
			stack2cycle(s+i,c+*n);
			stack2cycle_inverse(s+j,c+*n);
			if(!trace_cycle(c+*n)){
				if(!cmp_cycles(c,*n)){
					(*n)++;
				}
			}
		}
	}
	return *n;
}
int cycle_even(cycle *c,int *n,path *first,path *second,stack_edge *s,branch *b){
	int i,j,k,l;
	k=store_stack(first,s,b);
	l=store_stack(second,s+k,b);
	for(i=0;i<k;i++){
		for(j=k;j<l+k;j++){
			(c+*n)->length=0;
			stack2cycle(s+i,c+*n);
			stack2cycle_inverse(s+j,c+*n);
			if(!trace_cycle(c+*n)){
				if(!cmp_cycles(c,*n)){
					(*n)++;
				}
			}
		}
	}
	return 0;
}
int cycle_from_set(matrix *m){
	int i;
	int t=m->nxor-m->size+1;	
	m->ncycles=0;
	for(i=0;(i<m->cset->size)&&(m->ncycles!=t);i++){
	//for(i=0;(i<m->cset->size);i++){
		if(m->cset->c[i]->length%2){
			cycle_even(m->c,&m->ncycles,m->cset->c[i]->first,m->cset->c[i]->second,m->s,m->b);
		}
		else{
			cycle_odd(m->c,&m->ncycles,m->cset->c[i]->first,m->s,m->b);
		}
	}
	return 0;
}
void write_eps(char *name,matrix *m){
	FILE *f;
	int i,j;
	f=open_file2(name,".eps","w");
	//Header
	fprintf(f,"%%!\n"
		"%%%%BoundingBox: 0 0 600 600\n"
		"/Helvetica-Bold findfont\n"
		"4 scalefont\n"
		"setfont\n"
		"/psx 400 def\n"
		"/psy 400 def\n"
		"/hx %0.3lf def\n"
		"/hy %0.3lf def\n"
		"/tpsx 600 psx sub 2 div def\n"
		"/tpsy 600 psy sub 2 div def\n"
		"/sigma{psx mul 400 div}def\n"
		"/bgbox{\n"
		"	newpath\n"
		"	0 0 moveto\n"
		"	0 psy lineto\n"
		"	psx psy lineto\n"
		"	psx 0 lineto\n"
		"	closepath\n"
		"	0.87 1.0 0.85 setrgbcolor\n"
		"	fill\n"
		"}def\n"
		"/hashbox{\n"
		"	newpath\n"
		"	0 0 moveto\n"
		"	0 hy sigma lineto\n"
		"	hx sigma hy sigma lineto\n"
		"	hx sigma 0 lineto\n"
		"	closepath\n"
		"	gsave\n"
		"	1.0 0.1 0.1 setrgbcolor\n"
		"	fill\n"
		"	grestore\n"
		"	1.0 0.0 0.0 setrgbcolor\n"
		"	0.1 setlinewidth\n"
		"	stroke\n"
		"}def\n"
		"/hashbox2{\n"
		"	newpath\n"
		"	0 0 moveto\n"
		"	0 hy sigma lineto\n"
		"	hx sigma hy sigma lineto\n"
		"	hx sigma 0 lineto\n"
		"	closepath\n"
		"	gsave\n"
		"	0.1 0.1 0.2 setrgbcolor\n"
		"	fill\n"
		"	grestore\n"
		"	1.0 1.0 1.0 setrgbcolor\n"
		"	0.1 setlinewidth\n"
		"	stroke\n"
		"}def\n"
		"tpsx tpsy translate\n"
		"bgbox\n"
		,400.0/m->size,400.0/m->size);
	for(i=0;i<m->size;i++){
		for(j=0;j<m->size;j++){
			if(i==j){
				fprintf(f,"gsave\n%d sigma hx mul %d sigma hy mul translate\nhashbox2\ngrestore\n",i,j);
			}
			if(m->graph[i][j]){
				fprintf(f,"gsave\n%d sigma hx mul %d sigma hy mul translate\nhashbox\ngrestore\n",i,j);
			}
		}
	}
}
void write_dist_eps(char *name,matrix *m){
	FILE *f;
	int i,j;
	f=open_file2(name,"_dist.eps","w");
	//Header
	fprintf(f,"%%!\n"
		"%%%%BoundingBox: 0 0 600 600\n"
		"/Helvetica-Bold findfont\n"
		"4 scalefont\n"
		"setfont\n"
		"/psx 400 def\n"
		"/psy 400 def\n"
		"/hx %0.3lf def\n"
		"/hy %0.3lf def\n"
		"/tpsx 600 psx sub 2 div def\n"
		"/tpsy 600 psy sub 2 div def\n"
		"/sigma{psx mul 400 div}def\n"
		"/bgbox{\n"
		"	newpath\n"
		"	0 0 moveto\n"
		"	0 psy lineto\n"
		"	psx psy lineto\n"
		"	psx 0 lineto\n"
		"	closepath\n"
		"	0.87 1.0 0.85 setrgbcolor\n"
		"	fill\n"
		"}def\n"
		"/hashbox{\n"
		"	newpath\n"
		"	0 0 moveto\n"
		"	0 hy sigma lineto\n"
		"	hx sigma hy sigma lineto\n"
		"	hx sigma 0 lineto\n"
		"	closepath\n"
		"	gsave\n"
		//"	0.9 0.1 0.1 setrgbcolor\n"
		"	fill\n"
		"	grestore\n"
		//"	0.9 0.1 0.1 setrgbcolor\n"
		"	0.1 setlinewidth\n"
		"	stroke\n"
		"}def\n"
		"/hashbox2{\n"
		"	newpath\n"
		"	0 0 moveto\n"
		"	0 hy sigma lineto\n"
		"	hx sigma hy sigma lineto\n"
		"	hx sigma 0 lineto\n"
		"	closepath\n"
		"	gsave\n"
		"	0.9 0.9 0.9 setrgbcolor\n"
		"	fill\n"
		"	grestore\n"
		"	0.9 0.9 0.9 setrgbcolor\n"
		"	0.1 setlinewidth\n"
		"	stroke\n"
		"}def\n"
		"tpsx tpsy translate\n"
		"bgbox\n"
		,400.0/m->size,400.0/m->size);
	for(i=0;i<m->size;i++){
		for(j=0;j<m->size;j++){
			if(i==j){
				fprintf(f,"gsave\n%d sigma hx mul %d sigma hy mul translate\nhashbox2\ngrestore\n",i,j);
			}
			if(m->dist[i][j]){
				fprintf(f,"gsave\n%d sigma hx mul %d sigma hy mul translate\n%lf %lf %lf setrgbcolor\nhashbox\ngrestore\n",i,j,m->dist[i][j]/8.0,0.1,0.1);
			}
		}
	}
}

int swap_ij_graph(matrix *m,int i,int j){
	int k,l;
	int tmp_i[m->size];
	int tmp_j[m->size];
	for(k=0;k<m->size;k++){
		tmp_i[k]=m->graph[i][k];
		tmp_j[k]=m->graph[j][k];
	}
	for(k=0;k<m->size;k++){
		m->graph[i][k]=tmp_j[k];
		m->graph[j][k]=tmp_i[k];
		m->graph[k][i]=tmp_j[k];
		m->graph[k][j]=tmp_i[k];
	}
	for(k=0;k<m->size;k++){
		if(m->graph[k][k]){
			for(l=0;l<m->size;l++){
				m->graph[i][l]=tmp_i[l];
				m->graph[j][l]=tmp_j[l];
				m->graph[l][i]=tmp_i[l];
				m->graph[l][j]=tmp_j[l];
			}
			return 1;
		}
	}
	return 0;
}
int rnd_swap_graph(matrix *m){
	int i,j;
	i=dsfmt_genrand_open_open(&dsfmt)*m->size;
	do{
		j=dsfmt_genrand_open_open(&dsfmt)*m->size;
	}while(i==j);
	swap_ij_graph(m,i,j);
	return 0;
}
int diag_dij_graph(matrix *m,int i){
	int k,d;
	for(k=0,d=0;k<m->size;k++){
		if(m->graph[i][k]){
			if(abs(k-i)>d){
				d=abs(k-i);
			}
		}
	}
	return d;
}
int find_max_diag_graph(matrix *m){
	int k,l=0,d=0,dk;
	for(k=0;k<m->size;k++){
		dk=diag_dij_graph(m,k);
		if(dk>d){
			d=dk;
			l=k;
		}
	}
	return l;
}
int rnd_diag_swap_graph(matrix *m){
	int i,j;
	int di_old,dj_old;
	int di_new,dj_new;
	i=dsfmt_genrand_open_open(&dsfmt)*m->size;
	do{
		j=dsfmt_genrand_open_open(&dsfmt)*m->size;
	}while(i==j);
	di_old=diag_dij_graph(m,i);
	dj_old=diag_dij_graph(m,j);

	swap_ij_graph(m,i,j);
	di_new=diag_dij_graph(m,i);
	dj_new=diag_dij_graph(m,j);

	if(di_new+dj_new>di_old+dj_old){
		swap_ij_graph(m,i,j);
	}
	return 0;
}
int rnd_find_swap_graph(matrix *m){
	int i,j;
	int di_old,dj_old;
	int di_new,dj_new;
	//i=dsfmt_genrand_open_open(&dsfmt)*m->size;
	i=find_max_diag_graph(m);
	do{
		j=dsfmt_genrand_open_open(&dsfmt)*m->size;
	}while(i==j);
	di_old=diag_dij_graph(m,i);
	dj_old=diag_dij_graph(m,j);

	swap_ij_graph(m,i,j);
	di_new=diag_dij_graph(m,i);
	dj_new=diag_dij_graph(m,j);

	if(di_new+dj_new>di_old+dj_old){
		swap_ij_graph(m,i,j);
	}
	return 0;
}
//main
int main(int argc __attribute__ ((unused)),char *argv[] __attribute__ ((unused))){
	matrix *m=read_matrix("new.matrix");
	dsfmt_init_gen_rand(&dsfmt,1013);
	for(int i=0;i<1000;i++){
		rnd_diag_swap_graph(m);
		rnd_find_swap_graph(m);
	}
	init_edges(m);
	graph2dist(m);

	//printf("diagd %d\n",diag_dij_graph(m,3));
	char name[256]="t";
	write_eps(name,m);
	//return 0;
	//print_graph(stdout,m);
	//print_dist(stdout,m);

	floydwarshall(m);
	//print_dist(stdout,m);
	cycle_path_store(m);

	printf("cset->size %d/%d %d %d nxor %d\n",m->cset->size,m->size*m->size,m->cset->c[0]->length,m->nedges,m->nxor);

	cycle_from_set(m);
	print_cycle(stdout,m->c,m->ncycles);
	printf("ncycles %d\n",m->ncycles);

	write_dist_eps(name,m);
	return 0;
}
