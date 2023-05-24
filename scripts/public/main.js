let classes = []
for (let i = 0; i < 44; i++) {
	classes[i] = "entity" + i;
}

for (const cls of classes) {
	let elements = document.getElementsByClassName(cls);
	for (const element of elements) {
		element.addEventListener('mouseover', function () {
			console.log("mouseover")
			for (const element of elements) {
				element.style.backgroundColor = "red";
			}
		})
		element.addEventListener('mouseout', function () {
			for (const element of elements) {
				element.style.backgroundColor = "white";
			}
		})
	}
}