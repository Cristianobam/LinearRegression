let px = [];
let py = [];

// f(x) := m*x + b

const m = tf.variable(tf.randomUniform([1],-1,1).asScalar());
const b = tf.variable(tf.randomUniform([1],-1,1).asScalar());

const predict = (x)=> tf.tensor1d(x).mul(m).add(b);
const loss = (y_predict,y_real) => tf.losses.meanSquaredError(tf.tensor1d(y_real),y_predict);

const lr = 0.05;
const optimizer = tf.train.sgd(lr);

function setup(){
    createCanvas(windowWidth, windowHeight);
}

function draw(){
    background(0);

    if (px.length>0){
        optimizer.minimize(()=>loss(predict(px),py));
        //console.log(loss(predict(px),py).print());
    }
   
    for (let i=0; i<px.length; i++){
        noStroke();
        fill(255,255,255,100);
        const x = map(px[i], -1, 1, 0, width);
        const y = map(py[i], -1, 1, 0, height);
        circle(x,y,10);
    }

    tf.tidy(()=>{
    const lineX = [-1,1];
    let lineY = predict(lineX).dataSync();
    
    
    let x1 = map(lineX[0], -1, 1, 0, width);
    let x2 = map(lineX[1], -1, 1, 0, width);
    let y1 = map(lineY[0], -1, 1, 0, height);
    let y2 = map(lineY[1], -1, 1, 0, height);

    push();
    stroke(255);
    strokeWeight(3);
    line(x1, y1, x2, y2);
    pop();
    });
}
    
function mousePressed(){
    const x = map(mouseX, 0, width, -1, 1);
    const y = map(mouseY, height, 0, 1, -1);
    px.push(x);
    py.push(y);
}

function keyPressed(){
    if (key==='c'){
        px = px.splice();
        py = py.splice();
    }
}

function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
  }