#include<algorithm>
#include<cstring>
#include<vector>
#include<cstdio>
#include<cmath>

using namespace std;

const double eps = 1e-8;
const double PI = 4 * atan2(1, 1);
const double INF = 1e16;
const int N = 100005;
inline int sgn(double x){
	if(fabs(x) < eps)	return 0;
	else if(x > 0)	return 1;
	else	return -1;
}
inline int cmp(double x, double y){ return sgn(x-y); }
double rand01(){ return rand() / (double)RAND_MAX; }
double randeps(){ return (rand01() - 0.5) * eps; }

//------------------------------------ Vector & Point ------------------------------------//
struct Vector{
	double x, y;
	Vector() {}
	Vector(double x, double y):x(x), y(y){}
	void read(){ scanf("%lf%lf", &x, &y); }
};
typedef Vector Point;
Vector operator + (Vector A, Vector B){ return Vector(A.x + B.x, A.y + B.y); }
Vector operator - (Vector A, Vector B){ return Vector(A.x - B.x, A.y - B.y); }
Vector operator * (double k, Vector A){ return Vector(k * A.x, k * A.y); }
Vector operator * (Vector A, double k){ return k * A; }
Vector operator / (Vector A, double k){ return Vector(A.x / k, A.y / k); }
bool operator < (const Vector &A, const Vector &B){
	return cmp(A.x, B.x) == 0 ? cmp(A.y, B.y) < 0 : cmp(A.x, B.x) < 0;
}
bool operator > (const Vector &A, const Vector &B){ return B < A; }
bool operator == (const Vector &A, const Vector &B){ return (cmp(A.x, B.x) == 0) && (cmp(A.y, B.y) == 0); }
bool operator != (const Vector &A, const Vector &B){ return !(A == B); }
// dot product
double operator * (Vector A, Vector B){ return A.x * B.x + A.y * B.y; }
// cross product
double operator ^ (Vector A, Vector B){ return A.x * B.y - A.y * B.x; }
double Length(Vector A){ return sqrt(A * A); }
// polar angle of vector A, in (-PI,PI]
double Angle(Vector A){ return atan2(A.y, A.x); }
// angle between two vectors, in [0,PI]
double Angle(Vector A, Vector B){ return atan2(fabs(A ^ B), A * B); }
// angle between two vectors, in (-PI, PI]
double signedAngle(Vector A, Vector B){
	double ang = Angle(A, B); if(sgn(A ^ B) < 0) ang *= -1; return ang;
}
// get the normal vector of A
Vector Normal(Vector A){ double L = Length(A); return Vector(-A.y/L, A.x/L); }
// test if vector(bc) is to the left of (ab)
bool ToTheLeft(Point A, Point B, Point C){ return sgn((B - A) ^ (C - B)) > 0; }
// test if vector B is to the left of vector A
bool ToTheLeft(Vector A, Vector B){ return sgn(A ^ B) > 0; }
double DistancePointToPoint(Point A, Point B){ return Length(A-B); }
//------------------------------------------------------------------------------//

//------------------------------------ Line ------------------------------------//
struct Line{
	Point p;
	Vector v;
	double ang; // angle of inclination (-PI, PI]
	Line() {}
	Line(Point p, Vector v):p(p), v(v){ ang = atan2(v.y, v.x); }
	Line(double a, double b, double c){ // ax + by + c = 0
		if(sgn(a) == 0)         p = Point(0, -c/b), v = Vector(1, 0);
		else if(sgn(b) == 0)    p = Point(-c/a, 0), v = Vector(0, 1);
		else                    p = Point(0, -c/b), v = Vector(-b, a);
	}
	Point getPoint(double t){ return p + v * t; }
	bool operator < (const Line &L) const{ return ang < L.ang; }
};
bool PointOnLine(Point p, Line l){ return sgn(l.v ^ (p-l.p)) == 0; }
bool PointOnRight(Point p, Line l){ return sgn(l.v ^ (p-l.p)) < 0; }
bool LineParallel(Line l1, Line l2){ return sgn(l1.v ^ l2.v) == 0; }
bool LineSame(Line l1, Line l2){ return LineParallel(l1, l2) && sgn((l1.p-l2.p) ^ l1.v) == 0; }
Point GetLineIntersection(Line l1, Line l2){
	Vector u = l1.p - l2.p;
	double t = (l2.v ^ u) / (l1.v ^ l2.v);
	return l1.p + l1.v * t;
}
Point PointLineProjection(Point p, Line l){ return l.p + l.v * ((l.v * (p - l.p)) / (l.v * l.v)); }
bool PointOnSegment(Point p, Point A, Point B){
	return sgn((p - A) * (p - B)) <= 0 && sgn((p - A) ^ (p - B)) == 0;
}
double DistancePointToLine(Point p, Line l){ return fabs(((p - l.p) ^ l.v) / Length(l.v)); }

//------------------------------------ Circle ------------------------------------//
struct Circle{
	Point p;
	double r;
	Circle() {}
	Circle(Point p, double r):p(p), r(r) {}
	Point getPoint(double alpha){
		return Point(p.x + cos(alpha) * r, p.y + sin(alpha) * r);
	}
};
void getLineCircleIntersection(Line L, Circle C, Point res[], int &resn){
	// resn is the number of intersecton points
	// intersection points are stored in res[]
	resn = 0;
	Point q = PointLineProjection(C.p, L);
	double d = DistancePointToPoint(C.p, q);
	if(cmp(d, C.r) > 0)   return;                           // separated
	else if(cmp(d, C.r) == 0){ res[++resn] = q; return; }   // tangent
	Vector u = L.v / Length(L.v);
	double dd = sqrt(C.r * C.r - d * d);
	res[++resn] = q - dd * u, res[++resn] = q + dd * u;     // intersected
}
double TriangleCircleIntersectionArea(Point A, Point B, double r){
	// Circle's center is O(0, 0), radius is r
	// The triangle is OAB
	double ra = sqrt(A*A), rb = sqrt(B*B);
	Point O(0, 0);
	Line AB = Line(A, B-A);
	if(cmp(ra, r) <= 0 && cmp(rb, r) <= 0)	return (A ^ B) / 2;
	else if(cmp(ra, r) >= 0 && cmp(rb, r) >= 0){
		double d = DistancePointToLine(O, AB);
		double theta = signedAngle(A, B);
		if(cmp(d, r) >= 0)	return theta * r * r / 2;
		else{
			Point H = GetLineIntersection(AB, Line(O, Normal(B-A)));
			if(PointOnSegment(H, A, B)){
				Point t[3]; int _t;
				getLineCircleIntersection(AB, Circle(O, r), t, _t);
				double phi = signedAngle(t[1], t[2]);
				return (theta + sin(phi) - phi) * r * r / 2;
			}
			else	return theta * r * r / 2;
		}
	}
	else{
		Point t[3]; int _t;
		getLineCircleIntersection(AB, Circle(O, r), t, _t);
		if(PointOnSegment(t[2], A, B))	t[1] = t[2];
		if(cmp(ra, r) <= 0)
			return signedAngle(t[1], B) * r * r / 2 + (A ^ t[1]) / 2;
		else
			return signedAngle(A, t[1]) * r * r / 2 + (t[1] ^ B) / 2;
	}
}
double PolygonCircleIntersectionArea(int n, Point p[], Circle C){
	// p[] is a simple polygon
	// ATT: result might be negative
	double res = 0;
	for(int i = 1; i <= n; i++)
		res += TriangleCircleIntersectionArea(p[i]-C.p, p[i%n+1]-C.p, C.r);
	return res;
}
//-----------------------------------------------------------------------------------------------//

int main(){
	freopen("evaluate_in.txt", "r", stdin);
	double r; scanf("%lf", &r);
	vector<double> ans;
	for(int i = 1; i <= 1325; i++){
		Point P[4];
		P[1].read(), P[2].read(), P[3].read();
		double area = fabs(PolygonCircleIntersectionArea(3, P, Circle(Point(0, 0), r)));
		ans.push_back(area);
	}
	for(auto &e : ans)
		printf("%.6f ", e);
	return 0;
}