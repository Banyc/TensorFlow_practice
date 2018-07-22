import tensorflow as tf 
import numpy as np 
import os

from tensorflow.examples.tutorials.mnist import input_data
import inference

REGULARIZATION_RATE = 0.01
BATCH_SIZE = 500
TRAINING_STEPS = 1000

DATABASE_PATH = "./fashion_mnist/fashion-mnist/data/fashion"
MODEL_PATH = "./fashion_mnist/codes/LeNet-5/model"
MODEL_NAME = "model.ckpt"

LEARING_RATE_BASE = 0.01
DECAY_RATE = 0.99


def train(mnist):
    x = tf.placeholder(
        tf.float32,
        [None,
        inference.IMAGE_SIZE,
        inference.IMAGE_SIZE,
        inference.NUM_CHANNELS],
        name="x-input"
    )
    # y_ = tf.placeholder(
    #     tf.float32,
    #     [None, inference.NUM_LABELS],
    #     name="y_-input",
    # )
    y_ = tf.placeholder(
        tf.int32,
        [None],
        name="y_-input",
    )
    global_step = tf.Variable(
        0, False, dtype=tf.int16
    )
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.inference(x, True, regularizer)

    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=tf.argmax(y_, 1), logits=y
    # )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_, logits=y
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(
        LEARING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        DECAY_RATE
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    # with tf.control_dependencies(train_step):
        # train_op = tf.no_op("train")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            MODEL_PATH
        )
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(
                sess,
                ckpt.model_checkpoint_path
            )
        else:
            tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(
                xs, 
                [-1, 
                inference.IMAGE_SIZE,
                inference.IMAGE_SIZE,
                inference.NUM_CHANNELS],
            )
            # reshaped_ys = np.reshape(
            #     ys,
            #     [-1, inference.NUM_LABELS]
            # )
            # loss_value, _ = sess.run((loss, train_op), {x: reshaped_xs, y_: ys})
            loss_value, _ = sess.run((loss, train_step), {x: reshaped_xs, y_: ys})
            print(i, global_step.eval(), loss_value)
            print()
            if i % 50 == 0:
                # print(i, global_step, loss_value)
                saver.save(
                    sess,
                    os.path.join(MODEL_PATH, MODEL_NAME),
                    global_step
                )
        saver.save(
            sess,
            os.path.join(MODEL_PATH, MODEL_NAME),
            global_step
        )


def main(argv=None):
    mnist = input_data.read_data_sets(DATABASE_PATH)
    train(mnist)
    return 0


if __name__ == "__main__":
    tf.app.run()

# [Running] python ".\fashion_mnist\codes\LeNet-5\train.py"
# 2018-07-20 23:18:55.042883: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-20 23:18:55.046773: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-20 23:18:55.050421: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-20 23:18:55.054500: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-20 23:18:55.058548: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-20 23:18:55.062589: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-20 23:18:55.067019: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
# 2018-07-20 23:18:55.070759: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
# Extracting ./fashion_mnist/fashion-mnist/data/fashion\train-images-idx3-ubyte.gz
# Extracting ./fashion_mnist/fashion-mnist/data/fashion\train-labels-idx1-ubyte.gz
# Extracting ./fashion_mnist/fashion-mnist/data/fashion\t10k-images-idx3-ubyte.gz
# Extracting ./fashion_mnist/fashion-mnist/data/fashion\t10k-labels-idx1-ubyte.gz
# 0 1 70.21211

# 1 2 67.30614

# 2 3 65.83109

# 3 4 65.22621

# 4 5 64.949135

# 5 6 64.810326

# 6 7 64.59001

# 7 8 64.523636

# 8 9 64.54313

# 9 10 64.38001

# 10 11 64.335014

# 11 12 64.31794

# 12 13 64.25412

# 13 14 64.211006

# 14 15 64.11338

# 15 16 64.10025

# 16 17 64.03941

# 17 18 64.10911

# 18 19 64.03539

# 19 20 63.962505

# 20 21 63.89374

# 21 22 63.879936

# 22 23 63.839516

# 23 24 63.86858

# 24 25 63.78057

# 25 26 63.81481

# 26 27 63.7559

# 27 28 63.656853

# 28 29 63.64908

# 29 30 63.595734

# 30 31 63.555542

# 31 32 63.584377

# 32 33 63.50558

# 33 34 63.433674

# 34 35 63.45736

# 35 36 63.418037

# 36 37 63.373444

# 37 38 63.462185

# 38 39 63.35604

# 39 40 63.33995

# 40 41 63.292717

# 41 42 63.322853

# 42 43 63.350548

# 43 44 63.236122

# 44 45 63.22263

# 45 46 63.21021

# 46 47 63.155872

# 47 48 63.25325

# 48 49 63.144325

# 49 50 63.07512

# 50 51 63.09737

# 51 52 63.16281

# 52 53 63.028526

# 53 54 63.058933

# 54 55 63.008827

# 55 56 63.043865

# 56 57 62.938587

# 57 58 62.93637

# 58 59 62.948414

# 59 60 62.895763

# 60 61 62.901184

# 61 62 62.95505

# 62 63 62.953365

# 63 64 62.83048

# 64 65 62.773766

# 65 66 62.801384

# 66 67 62.81365

# 67 68 62.724537

# 68 69 62.745518

# 69 70 62.746586

# 70 71 62.75604

# 71 72 62.67466

# 72 73 62.721325

# 73 74 62.631832

# 74 75 62.589474

# 75 76 62.656628

# 76 77 62.697453

# 77 78 62.560658

# 78 79 62.61007

# 79 80 62.519043

# 80 81 62.591022

# 81 82 62.610847

# 82 83 62.633884

# 83 84 62.45869

# 84 85 62.498318

# 85 86 62.526485

# 86 87 62.436974

# 87 88 62.50079

# 88 89 62.449425

# 89 90 62.36266

# 90 91 62.40332

# 91 92 62.323906

# 92 93 62.330353

# 93 94 62.36987

# 94 95 62.260014

# 95 96 62.395653

# 96 97 62.28216

# 97 98 62.28073

# 98 99 62.22812

# 99 100 62.18028

# 100 101 62.22016

# 101 102 62.246883

# 102 103 62.18576

# 103 104 62.229

# 104 105 62.222263

# 105 106 62.149643

# 106 107 62.180878

# 107 108 62.069878

# 108 109 62.056988

# 109 110 62.10037

# 110 111 62.08558

# 111 112 62.049377

# 112 113 62.006504

# 113 114 62.007175

# 114 115 61.96645

# 115 116 62.02927

# 116 117 62.031895

# 117 118 61.97399

# 118 119 61.888954

# 119 120 61.89234

# 120 121 61.983047

# 121 122 61.978657

# 122 123 61.859295

# 123 124 61.936047

# 124 125 61.816376

# 125 126 61.80182

# 126 127 61.78675

# 127 128 61.845512

# 128 129 61.784435

# 129 130 61.840065

# 130 131 61.765114

# 131 132 61.77044

# 132 133 61.779102

# 133 134 61.75382

# 134 135 61.701477

# 135 136 61.657066

# 136 137 61.713444

# 137 138 61.62624

# 138 139 61.69042

# 139 140 61.53257

# 140 141 61.579216

# 141 142 61.617138

# 142 143 61.612797

# 143 144 61.6096

# 144 145 61.566555

# 145 146 61.568054

# 146 147 61.58313

# 147 148 61.52672

# 148 149 61.480957

# 149 150 61.52627

# 150 151 61.472797

# 151 152 61.50106

# 152 153 61.399952

# 153 154 61.398872

# 154 155 61.476154

# 155 156 61.53704

# 156 157 61.32095

# 157 158 61.423347

# 158 159 61.323265

# 159 160 61.37904

# 160 161 61.354664

# 161 162 61.343273

# 162 163 61.36123

# 163 164 61.276424

# 164 165 61.26514

# 165 166 61.233074

# 166 167 61.134407

# 167 168 61.245632

# 168 169 61.20371

# 169 170 61.161186

# 170 171 61.202194

# 171 172 61.153866

# 172 173 61.185165

# 173 174 61.189205

# 174 175 61.164898

# 175 176 61.140865

# 176 177 61.10433

# 177 178 61.03907

# 178 179 61.18376

# 179 180 61.153545

# 180 181 61.064587

# 181 182 60.999878

# 182 183 61.01935

# 183 184 61.083282

# 184 185 61.072975

# 185 186 61.02974

# 186 187 60.9813

# 187 188 61.021816

# 188 189 60.991516

# 189 190 60.89108

# 190 191 60.960648

# 191 192 60.99037

# 192 193 60.932083

# 193 194 60.866787

# 194 195 60.936066

# 195 196 60.82083

# 196 197 60.823418

# 197 198 60.832397

# 198 199 60.82016

# 199 200 60.941742

# 200 201 60.837105

# 201 202 60.846928

# 202 203 60.83941

# 203 204 60.783318

# 204 205 60.75482

# 205 206 60.6827

# 206 207 60.708504

# 207 208 60.71135

# 208 209 60.638885

# 209 210 60.710228

# 210 211 60.657684

# 211 212 60.70442

# 212 213 60.685734

# 213 214 60.64643

# 214 215 60.602745

# 215 216 60.599342

# 216 217 60.579098

# 217 218 60.673046

# 218 219 60.61944

# 219 220 60.625084

# 220 221 60.555637

# 221 222 60.507755

# 222 223 60.456955

# 223 224 60.51521

# 224 225 60.47899

# 225 226 60.43845

# 226 227 60.44904

# 227 228 60.454113

# 228 229 60.457146

# 229 230 60.429802

# 230 231 60.395557

# 231 232 60.388058

# 232 233 60.34123

# 233 234 60.31357

# 234 235 60.360916

# 235 236 60.390118

# 236 237 60.367508

# 237 238 60.417904

# 238 239 60.276497

# 239 240 60.26141

# 240 241 60.334362

# 241 242 60.25446

# 242 243 60.3072

# 243 244 60.2262

# 244 245 60.303352

# 245 246 60.21372

# 246 247 60.313194

# 247 248 60.215492

# 248 249 60.13184

# 249 250 60.280384

# 250 251 60.129967

# 251 252 60.1803

# 252 253 60.135532

# 253 254 60.1674

# 254 255 60.116524

# 255 256 60.046227

# 256 257 60.041492

# 257 258 60.02463

# 258 259 60.040096

# 259 260 60.02611

# 260 261 60.053646

# 261 262 59.97137

# 262 263 59.966816

# 263 264 60.0737

# 264 265 60.011856

# 265 266 60.059105

# 266 267 60.060204

# 267 268 59.911064

# 268 269 59.93564

# 269 270 59.90864

# 270 271 59.987236

# 271 272 59.98647

# 272 273 59.938725

# 273 274 59.822315

# 274 275 59.88236

# 275 276 59.83142

# 276 277 59.87546

# 277 278 59.813217

# 278 279 59.87402

# 279 280 59.82255

# 280 281 59.77393

# 281 282 59.790073

# 282 283 59.70379

# 283 284 59.728474

# 284 285 59.762146

# 285 286 59.7209

# 286 287 59.748924

# 287 288 59.67476

# 288 289 59.73269

# 289 290 59.69128

# 290 291 59.701534

# 291 292 59.67891

# 292 293 59.69589

# 293 294 59.683403

# 294 295 59.6013

# 295 296 59.59896

# 296 297 59.56622

# 297 298 59.702805

# 298 299 59.61409

# 299 300 59.635056

# 300 301 59.482536

# 301 302 59.55872

# 302 303 59.496696

# 303 304 59.526264

# 304 305 59.482533

# 305 306 59.462063

# 306 307 59.483566

# 307 308 59.4091

# 308 309 59.40063

# 309 310 59.45154

# 310 311 59.50949

# 311 312 59.426247

# 312 313 59.387062

# 313 314 59.436447

# 314 315 59.389557

# 315 316 59.382317

# 316 317 59.3325

# 317 318 59.350327

# 318 319 59.374195

# 319 320 59.298073

# 320 321 59.297394

# 321 322 59.324017

# 322 323 59.193527

# 323 324 59.29743

# 324 325 59.323517

# 325 326 59.157394

# 326 327 59.213394

# 327 328 59.225098

# 328 329 59.236046

# 329 330 59.143024

# 330 331 59.155594

# 331 332 59.167786

# 332 333 59.122326

# 333 334 59.183323

# 334 335 59.09746

# 335 336 59.14578

# 336 337 59.121323

# 337 338 59.11198

# 338 339 59.074234

# 339 340 59.0609

# 340 341 59.092438

# 341 342 58.957405

# 342 343 59.03006

# 343 344 59.02119

# 344 345 59.022114

# 345 346 59.1029

# 346 347 58.943287

# 347 348 59.00649

# 348 349 58.956337

# 349 350 59.01997

# 350 351 58.957577

# 351 352 59.047527

# 352 353 58.899338

# 353 354 58.85932

# 354 355 58.856827

# 355 356 58.867558

# 356 357 58.813755

# 357 358 58.801163

# 358 359 58.832157

# 359 360 58.870358

# 360 361 58.76898

# 361 362 58.770184

# 362 363 58.833237

# 363 364 58.851593

# 364 365 58.80203

# 365 366 58.825348

# 366 367 58.80717

# 367 368 58.75981

# 368 369 58.74958

# 369 370 58.726498

# 370 371 58.666405

# 371 372 58.70231

# 372 373 58.68779

# 373 374 58.632473

# 374 375 58.743595

# 375 376 58.62385

# 376 377 58.62308

# 377 378 58.62765

# 378 379 58.611774

# 379 380 58.573437

# 380 381 58.524452

# 381 382 58.622246

# 382 383 58.480656

# 383 384 58.616188

# 384 385 58.576984

# 385 386 58.484444

# 386 387 58.478867

# 387 388 58.546616

# 388 389 58.52431

# 389 390 58.47032

# 390 391 58.48101

# 391 392 58.46699

# 392 393 58.426544

# 393 394 58.45725

# 394 395 58.415405

# 395 396 58.425964

# 396 397 58.443893

# 397 398 58.318993

# 398 399 58.356503

# 399 400 58.395866

# 400 401 58.467884

# 401 402 58.359592

# 402 403 58.381153

# 403 404 58.28601

# 404 405 58.40592

# 405 406 58.26938

# 406 407 58.322754

# 407 408 58.281086

# 408 409 58.26494

# 409 410 58.234978

# 410 411 58.22152

# 411 412 58.2701

# 412 413 58.182655

# 413 414 58.2377

# 414 415 58.27531

# 415 416 58.0976

# 416 417 58.14141

# 417 418 58.13716

# 418 419 58.109116

# 419 420 58.187134

# 420 421 58.14049

# 421 422 58.153606

# 422 423 58.127678

# 423 424 58.18463

# 424 425 58.03656

# 425 426 58.13522

# 426 427 58.080883

# 427 428 57.98364

# 428 429 57.968914

# 429 430 58.058846

# 430 431 58.016514

# 431 432 58.048412

# 432 433 58.048237

# 433 434 57.975758

# 434 435 57.995842

# 435 436 57.991512

# 436 437 57.97862

# 437 438 57.96084

# 438 439 57.861492

# 439 440 57.959084

# 440 441 57.96764

# 441 442 57.87977

# 442 443 57.831654

# 443 444 57.91129

# 444 445 57.80797

# 445 446 57.81435

# 446 447 57.85748

# 447 448 57.82786

# 448 449 57.815445

# 449 450 57.80324

# 450 451 57.827705

# 451 452 57.77162

# 452 453 57.81833

# 453 454 57.687386

# 454 455 57.715668

# 455 456 57.718693

# 456 457 57.78958

# 457 458 57.67074

# 458 459 57.666435

# 459 460 57.70485

# 460 461 57.660027

# 461 462 57.737427

# 462 463 57.639805

# 463 464 57.66996

# 464 465 57.61792

# 465 466 57.68434

# 466 467 57.64349

# 467 468 57.5491

# 468 469 57.53821

# 469 470 57.505795

# 470 471 57.609383

# 471 472 57.551178

# 472 473 57.522835

# 473 474 57.57853

# 474 475 57.511467

# 475 476 57.48144

# 476 477 57.488583

# 477 478 57.51813

# 478 479 57.44997

# 479 480 57.429024

# 480 481 57.436676

# 481 482 57.426517

# 482 483 57.37114

# 483 484 57.453472

# 484 485 57.358444

# 485 486 57.390293

# 486 487 57.4393

# 487 488 57.38604

# 488 489 57.363674

# 489 490 57.293636

# 490 491 57.37289

# 491 492 57.316826

# 492 493 57.305264

# 493 494 57.40279

# 494 495 57.326344

# 495 496 57.310333

# 496 497 57.272953

# 497 498 57.253273

# 498 499 57.207024

# 499 500 57.204586

# 500 501 57.192795

# 501 502 57.188038

# 502 503 57.2284

# 503 504 57.160667

# 504 505 57.179832

# 505 506 57.241158

# 506 507 57.132034

# 507 508 57.076126

# 508 509 57.128086

# 509 510 57.09906

# 510 511 57.024586

# 511 512 57.160053

# 512 513 57.113846

# 513 514 57.07333

# 514 515 57.059757

# 515 516 57.037243

# 516 517 57.061337

# 517 518 57.12597

# 518 519 57.042355

# 519 520 57.07564

# 520 521 57.0077

# 521 522 56.97005

# 522 523 56.96137

# 523 524 56.984776

# 524 525 56.933952

# 525 526 56.93054

# 526 527 56.976242

# 527 528 56.937065

# 528 529 56.927307

# 529 530 56.826515

# 530 531 56.83882

# 531 532 56.846283

# 532 533 56.841812

# 533 534 56.845257

# 534 535 56.825287

# 535 536 56.87171

# 536 537 56.86407

# 537 538 56.83852

# 538 539 56.76259

# 539 540 56.76508

# 540 541 56.84595

# 541 542 56.87355

# 542 543 56.791447

# 543 544 56.745567

# 544 545 56.755463

# 545 546 56.728287

# 546 547 56.765335

# 547 548 56.650665

# 548 549 56.639156

# 549 550 56.72436

# 550 551 56.68069

# 551 552 56.734932

# 552 553 56.665897

# 553 554 56.680637

# 554 555 56.61157

# 555 556 56.640594

# 556 557 56.655895

# 557 558 56.60586

# 558 559 56.482513

# 559 560 56.49032

# 560 561 56.54512

# 561 562 56.542397

# 562 563 56.551685

# 563 564 56.440533

# 564 565 56.50968

# 565 566 56.574017

# 566 567 56.539925

# 567 568 56.488213

# 568 569 56.510906

# 569 570 56.44298

# 570 571 56.386444

# 571 572 56.42182

# 572 573 56.376698

# 573 574 56.4365

# 574 575 56.415825

# 575 576 56.443214

# 576 577 56.318905

# 577 578 56.37731

# 578 579 56.380737

# 579 580 56.33791

# 580 581 56.25134

# 581 582 56.316998

# 582 583 56.35152

# 583 584 56.2938

# 584 585 56.33579

# 585 586 56.26973

# 586 587 56.319416

# 587 588 56.315105

# 588 589 56.27645

# 589 590 56.256252

# 590 591 56.25184

# 591 592 56.22982

# 592 593 56.23472

# 593 594 56.18219

# 594 595 56.16145

# 595 596 56.19333

# 596 597 56.178844

# 597 598 56.25296

# 598 599 56.208496

# 599 600 56.162613

# 600 601 56.15656

# 601 602 56.19337

# 602 603 56.08699

# 603 604 56.143448

# 604 605 56.132217

# 605 606 56.055172

# 606 607 55.980213

# 607 608 56.007465

# 608 609 56.013172

# 609 610 56.017635

# 610 611 56.035057

# 611 612 56.049725

# 612 613 56.029335

# 613 614 55.974747

# 614 615 56.00309

# 615 616 55.998158

# 616 617 55.889256

# 617 618 55.983852

# 618 619 55.90995

# 619 620 55.96105

# 620 621 55.934216

# 621 622 55.9282

# 622 623 55.865906

# 623 624 55.916924

# 624 625 55.936615

# 625 626 55.864525

# 626 627 55.806793

# 627 628 55.878216

# 628 629 55.898212

# 629 630 55.818737

# 630 631 55.809128

# 631 632 55.766487

# 632 633 55.737972

# 633 634 55.744556

# 634 635 55.787273

# 635 636 55.647484

# 636 637 55.720947

# 637 638 55.74639

# 638 639 55.700233

# 639 640 55.66288

# 640 641 55.68371

# 641 642 55.611282

# 642 643 55.66876

# 643 644 55.687622

# 644 645 55.711437

# 645 646 55.66673

# 646 647 55.686813

# 647 648 55.649254

# 648 649 55.601967

# 649 650 55.539547

# 650 651 55.534916

# 651 652 55.606045

# 652 653 55.583916

# 653 654 55.528122

# 654 655 55.60218

# 655 656 55.487053

# 656 657 55.580204

# 657 658 55.513706

# 658 659 55.521595

# 659 660 55.45906

# 660 661 55.532475

# 661 662 55.44555

# 662 663 55.458935

# 663 664 55.4407

# 664 665 55.412163

# 665 666 55.36722

# 666 667 55.46312

# 667 668 55.424953

# 668 669 55.358475

# 669 670 55.484737

# 670 671 55.456932

# 671 672 55.39359

# 672 673 55.35339

# 673 674 55.332615

# 674 675 55.35119

# 675 676 55.34523

# 676 677 55.285065

# 677 678 55.287304

# 678 679 55.331673

# 679 680 55.25019

# 680 681 55.25884

# 681 682 55.231407

# 682 683 55.28856

# 683 684 55.26212

# 684 685 55.244442

# 685 686 55.293945

# 686 687 55.174694

# 687 688 55.26509

# 688 689 55.2985

# 689 690 55.130795

# 690 691 55.134544

# 691 692 55.172264

# 692 693 55.13494

# 693 694 55.13803

# 694 695 55.048477

# 695 696 55.17529

# 696 697 55.097763

# 697 698 55.069218

# 698 699 55.12339

# 699 700 55.03496

# 700 701 55.121593

# 701 702 55.068756

# 702 703 55.09834

# 703 704 54.973804

# 704 705 55.05061

# 705 706 55.032646

# 706 707 54.992332

# 707 708 54.969696

# 708 709 55.009235

# 709 710 54.959816

# 710 711 54.932426

# 711 712 54.961777

# 712 713 54.96553

# 713 714 54.92253

# 714 715 54.954727

# 715 716 54.931435

# 716 717 54.87389

# 717 718 54.929558

# 718 719 54.921127

# 719 720 54.890224

# 720 721 54.82508

# 721 722 54.833656

# 722 723 54.867973

# 723 724 54.85727

# 724 725 54.801643

# 725 726 54.742245

# 726 727 54.75848

# 727 728 54.7921

# 728 729 54.81799

# 729 730 54.757664

# 730 731 54.713654

# 731 732 54.764294

# 732 733 54.731255

# 733 734 54.748528

# 734 735 54.70882

# 735 736 54.652287

# 736 737 54.69368

# 737 738 54.70802

# 738 739 54.67109

# 739 740 54.683514

# 740 741 54.66431

# 741 742 54.627666

# 742 743 54.62918

# 743 744 54.563995

# 744 745 54.642242

# 745 746 54.618015

# 746 747 54.617435

# 747 748 54.543865

# 748 749 54.5933

# 749 750 54.511803

# 750 751 54.59681

# 751 752 54.55435

# 752 753 54.56957

# 753 754 54.589245

# 754 755 54.566696

# 755 756 54.470913

# 756 757 54.514

# 757 758 54.598015

# 758 759 54.464962

# 759 760 54.42724

# 760 761 54.433548

# 761 762 54.453785

# 762 763 54.413307

# 763 764 54.40162

# 764 765 54.372738

# 765 766 54.36952

# 766 767 54.352856

# 767 768 54.353287

# 768 769 54.345394

# 769 770 54.355534

# 770 771 54.363335

# 771 772 54.323837

# 772 773 54.326965

# 773 774 54.30988

# 774 775 54.266663

# 775 776 54.315273

# 776 777 54.290062

# 777 778 54.247166

# 778 779 54.25464

# 779 780 54.269573

# 780 781 54.191814

# 781 782 54.271523

# 782 783 54.20743

# 783 784 54.159184

# 784 785 54.250957

# 785 786 54.145954

# 786 787 54.240086

# 787 788 54.25224

# 788 789 54.224575

# 789 790 54.16961

# 790 791 54.200798

# 791 792 54.124D:\ProgramData\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
#   from ._conv import register_converters as _register_converters


# 792 793 54.14136

# 793 794 54.059044

# 794 795 54.049316

# 795 796 54.06948

# 796 797 54.047813

# 797 798 54.095543

# 798 799 54.07804

# 799 800 54.05049

# 800 801 54.01468

# 801 802 54.077183

# 802 803 53.974487

# 803 804 54.013096

# 804 805 54.049286

# 805 806 53.988026

# 806 807 54.00597

# 807 808 53.975994

# 808 809 53.955055

# 809 810 53.939476

# 810 811 53.924305

# 811 812 53.898182

# 812 813 53.89458

# 813 814 53.905815

# 814 815 53.85163

# 815 816 53.922726

# 816 817 53.87816

# 817 818 53.85864

# 818 819 53.828114

# 819 820 53.89675

# 820 821 53.865746

# 821 822 53.85372

# 822 823 53.82462

# 823 824 53.796383

# 824 825 53.83002

# 825 826 53.766506

# 826 827 53.778236

# 827 828 53.751347

# 828 829 53.75878

# 829 830 53.784676

# 830 831 53.70608

# 831 832 53.703667

# 832 833 53.739704

# 833 834 53.687096

# 834 835 53.763275

# 835 836 53.694138

# 836 837 53.62941

# 837 838 53.62713

# 838 839 53.662273

# 839 840 53.66977

# 840 841 53.6314

# 841 842 53.64756

# 842 843 53.677708

# 843 844 53.596397

# 844 845 53.639088

# 845 846 53.582584

# 846 847 53.559834

# 847 848 53.67119

# 848 849 53.55905

# 849 850 53.589264

# 850 851 53.4802

# 851 852 53.55904

# 852 853 53.49429

# 853 854 53.52685

# 854 855 53.579395

# 855 856 53.480034

# 856 857 53.457264

# 857 858 53.481197

# 858 859 53.46042

# 859 860 53.40966

# 860 861 53.466026

# 861 862 53.446457

# 862 863 53.4041

# 863 864 53.404636

# 864 865 53.391285

# 865 866 53.34392

# 866 867 53.39873

# 867 868 53.361008

# 868 869 53.389103

# 869 870 53.394314

# 870 871 53.287323

# 871 872 53.31893

# 872 873 53.270638

# 873 874 53.304058

# 874 875 53.375374

# 875 876 53.28398

# 876 877 53.324276

# 877 878 53.283543

# 878 879 53.256138

# 879 880 53.235863

# 880 881 53.2472

# 881 882 53.212585

# 882 883 53.2417

# 883 884 53.22108

# 884 885 53.253056

# 885 886 53.276527

# 886 887 53.127075

# 887 888 53.178207

# 888 889 53.173744

# 889 890 53.264175

# 890 891 53.14033

# 891 892 53.15162

# 892 893 53.11681

# 893 894 53.04964

# 894 895 53.106606

# 895 896 53.094246

# 896 897 53.10832

# 897 898 53.014854

# 898 899 53.04802

# 899 900 52.992958

# 900 901 53.062637

# 901 902 53.032486

# 902 903 53.061954

# 903 904 53.04152

# 904 905 53.043583

# 905 906 52.950363

# 906 907 53.003372

# 907 908 52.963314

# 908 909 52.906364

# 909 910 52.97291

# 910 911 52.903862

# 911 912 53.007397

# 912 913 52.910492

# 913 914 52.911007

# 914 915 52.840504

# 915 916 52.928677

# 916 917 52.869514

# 917 918 52.89192

# 918 919 52.83498

# 919 920 52.789227

# 920 921 52.893124

# 921 922 52.84613

# 922 923 52.795826

# 923 924 52.8529

# 924 925 52.805767

# 925 926 52.915962

# 926 927 52.807045

# 927 928 52.74644

# 928 929 52.803288

# 929 930 52.875416

# 930 931 52.73163

# 931 932 52.679913

# 932 933 52.753674

# 933 934 52.740864

# 934 935 52.708553

# 935 936 52.733646

# 936 937 52.70313

# 937 938 52.708782

# 938 939 52.61796

# 939 940 52.69504

# 940 941 52.639248

# 941 942 52.611214

# 942 943 52.59289

# 943 944 52.66126

# 944 945 52.643658

# 945 946 52.65564

# 946 947 52.59675

# 947 948 52.57606

# 948 949 52.49952

# 949 950 52.565197

# 950 951 52.53586

# 951 952 52.5913

# 952 953 52.496674

# 953 954 52.586246

# 954 955 52.47624

# 955 956 52.45131

# 956 957 52.530903

# 957 958 52.526115

# 958 959 52.472824

# 959 960 52.427334

# 960 961 52.467934

# 961 962 52.45393

# 962 963 52.511894

# 963 964 52.440273

# 964 965 52.427834

# 965 966 52.412098

# 966 967 52.461544

# 967 968 52.438168

# 968 969 52.354454

# 969 970 52.35737

# 970 971 52.40094

# 971 972 52.366386

# 972 973 52.314545

# 973 974 52.370045

# 974 975 52.393017

# 975 976 52.278595

# 976 977 52.342964

# 977 978 52.28071

# 978 979 52.3434

# 979 980 52.26211

# 980 981 52.264755

# 981 982 52.29251

# 982 983 52.23676

# 983 984 52.242157

# 984 985 52.239536

# 985 986 52.197227

# 986 987 52.215073

# 987 988 52.146214

# 988 989 52.193092

# 989 990 52.193867

# 990 991 52.154068

# 991 992 52.269608

# 992 993 52.173374

# 993 994 52.173786

# 994 995 52.225697

# 995 996 52.14316

# 996 997 52.226006

# 997 998 52.122383

# 998 999 52.17654

# 999 1000 51.98497