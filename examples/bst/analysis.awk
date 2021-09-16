{
	if ($2 > 0) {
		s++
	}
	l++
	if (l == n) {
		print s / l
		s = l = 0
	}
}
END {
	if (l > 0) {
		print s / l
	}
}
