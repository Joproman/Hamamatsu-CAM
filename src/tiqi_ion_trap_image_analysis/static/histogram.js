function tweakHistogram() {
    var min = $(".min").val();
    var max = $(".max").val();
    $.ajax({
        url: "/tweak_hist",
        type: "get",
        data: {
            'min': min,
            'max': max
        }
    });
}

$('.exp-click').on('click', tweakHistogram);