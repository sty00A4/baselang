var PI = 3.141592653589793
func floor(x): x // 1
func ceil(x): if x - x // 1 == 0 then x else x // 1 + 1
func round(x): if x - x // 1 >= 0.5 then ceil(x) else floor(x)
func abs(x): if x < 0 then -x else x