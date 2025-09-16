function changeExpTime() {
    var expTime = $(".newExpTime").val();
    $.ajax({
        url: "/set_exposure_time",
        type: "get",
        data: {newExp: expTime},
        success: function(response) {
            $(".expTime span").text(response);
        },
        error: function(jqxhr, status, exception) {
            alert('Exception:', exception);
        }
    });
}

$('.newExpTime').on('keypress', function (e) {
    if (e.which === 13){
        changeExpTime()
    }
})