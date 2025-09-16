function rotate() {
    var angle = $(".rotationAngle").text();
    $.ajax({
        url: "/rotate",
        type: "get",
        data: {'angle': angle},
        success: function(response) {
            $(".rotationAngle").text(response);
            document.getElementById("rotateButton").value = `Rotate image (${response})`
        },
        error: function(jqxhr, status, exception) {
            alert('Exception:', exception);
        }
    });
}